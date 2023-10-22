import torch
import numpy as np
import pickle

class LJ(torch.nn.Module):
    """ 
    torchのautogradを用いて, virialを計算するクラス
    """
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cell: torch.Tensor,
        cut_off: torch.Tensor,
    ):
        pos.requires_grad_(True)
        cell = torch.diag(cell)

        strain = torch.zeros_like(cell)       
        strain.requires_grad_(True)
        symmetric_strain =  0.5 * (strain + strain.transpose(-1, -2))

        cell = cell + torch.matmul(
            cell, symmetric_strain
        )

        pos = pos + torch.matmul(
            pos[:, None, :], symmetric_strain
        ).squeeze(1)

        vector_i_to_j = pos[edge_index[1]] - pos[edge_index[0]]

        offsets = torch.zeros((edge_index.shape[0], 3))
        offsets = -1 * (vector_i_to_j > cut_off) * torch.diag(cell,0)
        offsets += (vector_i_to_j < -cut_off) * torch.diag(cell,0)
        
        # 非対角成分のみ
        offsets = offsets + torch.matmul(
            offsets, symmetric_strain - torch.diag(symmetric_strain,0).diag()
        )

        vector_i_to_j += offsets

        r2 = vector_i_to_j.pow(2).sum(dim=1)
        assert (r2 < cut_off * cut_off).all()
        inv_r2 = 1.0 / r2
        C0 = -4.0 * (1/cut_off**12 - 1/cut_off**6) 

        total_energy = torch.sum(
            4 * (inv_r2**6 - inv_r2**3) + C0
        )

        grads = torch.autograd.grad(
            [
                -1 * total_energy,
            ],
            [
                pos, strain
            ],
            grad_outputs = torch.ones_like(total_energy),
            retain_graph=self.training,
            create_graph=self.training,
        )
        force = grads[0]
        if force is None:
            assert False, "force is not calculated"

        virial = grads[1]
        if virial is None:
            assert False, "virial is not calculated"
        
        outputs = {
            "virial": virial,
            "energy": total_energy,
            "force" : force,
        }
        return outputs

def GetVirial(pos,
              edge_index,
              cell,
              cut_off):
    """
    力と座標変位から、virialを求める
    """
    Energy = 0.0
    virial = np.zeros(6)
    C0 = -4.0 * (1/cut_off**12 - 1/cut_off**6)
    f = [0.0 for i in range(len(edge_index[0]))]
    for i in range(len(edge_index[0])):
        d = pos[edge_index[1][i]] - pos[edge_index[0][i]]
        d -= (d > cut_off) * cell
        d += (d < -cut_off) * cell

        r2 = np.power(d ,2).sum()
        assert r2 < cut_off * cut_off
        inv_r2 = 1.0 / r2
        Energy += 4.0 * (inv_r2**6 - inv_r2**3) + C0
        force = (24.0 * inv_r2**3 - 48.0 * inv_r2**6) * inv_r2
        f[i] += force
        virial[0] -= force * d[0] * d[0]
        virial[1] -= force * d[1] * d[1]
        virial[2] -= force * d[2] * d[2]
        virial[3] -= force * d[0] * d[1]
        virial[4] -= force * d[1] * d[2]
        virial[5] -= force * d[2] * d[0]

    return Energy, virial

def tch_virial_to_np(v: torch.Tensor):
    v = v.to('cpu').detach().numpy().copy()
    virial = np.zeros(6)
    virial[0] = v[0][0]
    virial[1] = v[1][1]
    virial[2] = v[2][2]
    virial[3] = v[0][1]
    virial[4] = v[1][2]
    virial[5] = v[2][0]
    return virial

# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# 二つの方法でvirialを計算する

# pickle data (pos, cell , edge_index, cut_off) path
dataset_path = "/nfshome17/knakajima/vasp_work/dataset/train_dataset/10_1_23/Ni_NPT_300K_1000step/Ni_NPT_300K_1000step_0.pickle"
model = LJ()

with open(dataset_path, "rb") as f:
    frames = pickle.load(f)

for frame_idx in range(len(frames)):
    # import data
    frames[frame_idx]["pos"] = torch.tensor(frames[frame_idx]["pos"], device="cpu", dtype=torch.float64) # (atomN, 3)
    frames[frame_idx]["edge_index"] = torch.tensor(frames[frame_idx]["edge_index"], device="cpu", dtype=torch.long) #(edgeN, 2)
    frames[frame_idx]["cell"] = torch.tensor(frames[frame_idx]["cell"], device="cpu", dtype=torch.float64) # (1,3)
    frames[frame_idx]["cut_off"] = torch.tensor(frames[frame_idx]["cut_off"], device="cpu", dtype=torch.float64) # (1)

    # get virial from torch.auto_grad
    outputs = model(
        frames[frame_idx]["pos"],
        frames[frame_idx]["edge_index"],
        frames[frame_idx]["cell"],
        frames[frame_idx]["cut_off"]
    )
    tch_virial = tch_virial_to_np(outputs["virial"])
    tch_energy = outputs["energy"].to('cpu').detach().numpy().copy()
    tch_force = outputs["force"].to("cpu").detach().numpy().copy()

    # get virial from F * d
    Fd_virial = np.zeros(6)
    Fd_energy = 0.0
    Fd_energy, Fd_virial = GetVirial(
        frames[frame_idx]["pos"].to('cpu').detach().numpy().copy(),
        frames[frame_idx]["edge_index"].to('cpu').detach().numpy().copy(),
        frames[frame_idx]["cell"].to('cpu').detach().numpy().copy(),
        frames[frame_idx]["cut_off"].to('cpu').detach().numpy().copy(),
    )
    np.set_printoptions(precision=4, floatmode='maxprec')

    # print result
    print(f"---{frame_idx}----------------------------------------------------------------------------------------------------------------------------", flush=True)
    print(f"method  Tot_E  XX  YY   ZZ   XY   YZ   ZX", flush=True)
    print(f"Torch {tch_energy} {tch_virial[0]} {tch_virial[1]} {tch_virial[2]} {tch_virial[3]} {tch_virial[4]} {tch_virial[5]}", flush=True)
    print(f" Fd   {Fd_energy} {Fd_virial[0]} {Fd_virial[1]} {Fd_virial[2]} {Fd_virial[3]} {Fd_virial[4]} {Fd_virial[5]}", flush=True)
    print("-------------------------------------------------------------------------------------------------------------------------------", flush=True)

