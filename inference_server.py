# inference_server.py
# 배치 추론용 별도 프로세스 (Manager 기반 요청/응답)
#
# 변경: reply_queue 객체를 요청으로 전달하지 않고,
#       req_q에 (req_id, planes) 넣고, 서버는 manager.dict()[req_id] = (logits, value)로 응답합니다.
#
import multiprocessing as mp
import time
import os
import torch
import numpy as np
from alphazero_model import AlphaZeroNet

class InferenceServer(mp.Process):
    def __init__(self, model_state_dict_path, req_q, res_dict, board_size=19,
                 device='cuda:0', batch_size=64, timeout=0.02, num_filters=128):
        """
        req_q: Manager().Queue() - receives tuples (req_id, planes) or ('__STOP__', None)
        res_dict: Manager().dict() - server writes results as res_dict[req_id] = (logits, value)
        """
        super().__init__()
        self.model_state_dict_path = model_state_dict_path
        self.board_size = board_size
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        self.num_filters = num_filters
        self.req_q = req_q
        self.res_dict = res_dict

    def run(self):
        torch_device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        model = AlphaZeroNet(board_size=self.board_size, num_filters=self.num_filters).to(torch_device)
        if self.model_state_dict_path and os.path.exists(self.model_state_dict_path):
            try:
                sd = torch.load(self.model_state_dict_path, map_location='cpu')
                model.load_state_dict(sd)
            except Exception as e:
                print("[InferenceServer] Warning: failed to load state_dict:", e)
        model.eval()

        while True:
            batch_planes = []
            req_ids = []
            t0 = time.time()
            # block for up to timeout to collect one request
            try:
                item = self.req_q.get(timeout=self.timeout)
            except Exception:
                continue
            # item expected: (req_id, planes) or ('__STOP__', None)
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            req_id, planes = item
            if req_id == "__STOP__":
                break
            batch_planes.append(planes)
            req_ids.append(req_id)

            # try to collect more up to batch_size
            while len(batch_planes) < self.batch_size:
                try:
                    item = self.req_q.get_nowait()
                except Exception:
                    break
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                rid, pl = item
                if rid == "__STOP__":
                    # put stop back for others and break
                    self.req_q.put(("__STOP__", None))
                    break
                batch_planes.append(pl)
                req_ids.append(rid)

            # prepare tensor and run model
            try:
                tensor = np.stack(batch_planes, axis=0).astype(np.float32)  # (B,2,H,W)
                t = torch.from_numpy(tensor).to(torch_device)
                with torch.no_grad():
                    logits, values = model(t)  # logits: (B, H*W), values: (B,)
                logits = logits.cpu().numpy()
                values = values.cpu().numpy()
            except Exception as e:
                # on error, write zero responses
                print("[InferenceServer] inference error:", e)
                for rid in req_ids:
                    try:
                        self.res_dict[rid] = (np.zeros(self.board_size*self.board_size, dtype=np.float32), 0.0)
                    except Exception:
                        pass
                continue

            # write back to manager.dict
            for i, rid in enumerate(req_ids):
                try:
                    self.res_dict[rid] = (logits[i], float(values[i]))
                except Exception:
                    pass

        # optional cleanup: mark a special key
        try:
            self.res_dict["__SERVER_STOP__"] = True
        except Exception:
            pass