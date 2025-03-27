from torch.autograd import gradcheck
import pytest
import torch
from mlwc.cpmd.pbc.pbc_torch import pbc_2d_torch, pbc_3d_torch, pbc_test, pbc_tutorial


def test_pbc_2d_torch():
    # モデルの初期化
    model = pbc_2d_torch()

    # 2D テストデータ作成 (ランダムな 3×3 セルとベクトル)
    torch.manual_seed(0)
    cell = torch.eye(3, dtype=torch.float32) * 5.0  # 単位セル（拡大）
    vectors_array = torch.randn(
        10, 3, dtype=torch.float32, requires_grad=True)  # 10個の3次元ベクトル

    # 期待される出力形状の確認
    output = model(vectors_array, cell)
    print(f"output.grad_fn = {output.grad_fn}")
    print(f"output.requires_grad = {output.requires_grad}")
    assert output.shape == vectors_array.shape, f"Expected shape {vectors_array.shape}, but got {output.shape}"

    # grad check
    vectors_array = vectors_array.clone().requires_grad_(True)
    cell = cell.clone()
    assert torch.autograd.gradgradcheck(
        model, (vectors_array, cell), eps=1e-3, atol=1e-4, rtol=1e-3), "Gradient check failed"


def test_pbc_3d_torch():
    model = pbc_3d_torch()
    model.eval()

    # 形状 [2, 3, 3] のベクトル配列
    vectors_array = torch.tensor([
        [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
        [[-0.1, -0.2, -0.3], [-1.1, -1.2, -1.3], [-2.1, -2.2, -2.3]],
    ], dtype=torch.float32, requires_grad=True)

    # 3×3 の単位格子
    cell = torch.eye(3, dtype=torch.float32, requires_grad=True)

    # 1. 基本的な動作確認
    pbc_vectors = model(vectors_array, cell)
    assert pbc_vectors.shape == vectors_array.shape

    # 2. 勾配計算が可能かどうかの確認
    assert gradcheck(model, (vectors_array.double(
    ).requires_grad_(), cell.double().requires_grad_()))

    # 3. エラー処理テスト
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 2), cell)  # 形状が (a, b, 3) でない場合

    with pytest.raises(ValueError):
        model(vectors_array, torch.randn(3, 2))  # cell の形状が (3, 3) でない場合


def test_export_torchscript():
    scripted_gate = torch.jit.script(pbc_tutorial())

    model = pbc_test()
    script = torch.jit.script(model)

    model = pbc_2d_torch()
    script = torch.jit.script(model)

    model = pbc_3d_torch()
    script = torch.jit.script(model)

    scripted_gate = torch.jit.script(pbc_tutorial())
