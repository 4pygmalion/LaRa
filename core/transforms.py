import torch


class TruncateOrPad:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def __call__(
        self,
        hpo_tensors: torch.Tensor,
    ) -> torch.Tensor:

        n_pad = self.max_len - len(hpo_tensors)
        if n_pad >= 1:
            return torch.nn.functional.pad(
                hpo_tensors, (0, 0, 0, n_pad), mode="constant", value=0
            )

        indices = torch.randperm(len(hpo_tensors))[: self.max_len]
        return hpo_tensors[indices]
