import os
import torch


def compile_siglip2_vit(
    model_name: str = "google/siglip2-giant-opt-patch16-256",
    save_path: str = "./compiled/vit/",
    input_shape: tuple = (1, 3, 256, 256),
) -> None:
    import torch_neuronx
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(model_name).eval()
    traced = torch_neuronx.trace(
        model,
        torch.randn(*input_shape),
        compiler_workdir=save_path + "workdir/",
        compiler_args=[
            "--auto-cast", "matmult",
            "--optlevel", "3",
            "--model-type", "unet-inference",
        ],
    )
    os.makedirs(save_path, exist_ok=True)
    torch.jit.save(traced, save_path + "model.pt")
