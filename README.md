This project builds upon PCP-MAE for the encoder pretraining. Original repository: https://github.com/aHapBean/PCP-MAE.

Force the model configuration to be consistent with MiniGPT-3D.

Adapt the PCP-MAE encoder using the Objaverse .npy dataset.

After modifications, pre-training with the ShapeNet55 dataset and using the original project's evaluation framework are no longer feasible. Currently, performance can only be evaluated by adopting MiniGPT-3D for 3D point cloud description as the downstream task.

AMP is enabled, supporting training on a single RTX 3090 graphics card, though the training cycle is relatively long. It is recommended to reduce the total number of epochs.

## Contact

If you have any questions related to the code or the paper, feel free to email Xiangdong (`zhangxiangdong@sjtu.edu.cn`) or Shaofeng (`sherrylone@sjtu.edu.cn`).

## License

PCP-MAE is released under MIT License. See the [LICENSE](./LICENSE) file for more details. Besides, the licensing information for `pointnet2` modules is available [here](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE).

## Acknowledgements

This codebase is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [ReCon](https://github.com/qizekun/ReCon), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2024pcp,
  title={PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders},
  author={Zhang, Xiangdong and Zhang, Shaofeng and Yan, Junchi},
  journal={arXiv preprint arXiv:2408.08753},
  year={2024}
}
```