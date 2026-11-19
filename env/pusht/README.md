# Push-T 仿真环境要求与已验证组合

当前验证平台：Linux x86_64、Ubuntu 24.04.5、GLIBC 2.39、Python 3.11.17、NVIDIA驱动570.86.10。与Lift保持依赖隔离；XPolicyLab服务端使用独立兼容环境。其他平台需单独验证。

| 组件 | 本次已安装版本 |
| --- | --- |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab core/assets/tasks/rl/mimic | 0.54.3 / 0.2.4 / 0.11.14 / 0.5.0 / 1.0.16 |
| torch / torchvision / torchaudio | 2.7.0+cu126 / 0.22.0+cu126 / 2.7.0 |
| NumPy / SciPy | 1.26.0 / 1.15.3 |
| FastAPI / Starlette | 0.115.7 / 0.45.3 |
| packaging / wheel / setuptools | 23.0 / 0.45.1 / 81.0.0 |
| Warp | 1.11.0 |
| CuRobo固定fork | d17b54ce32cba095c0b000c4c58777075d11de0e；包版本0.0.post1.dev100 |
| CUDA Core / CUDA Python绑定 | 0.7.0 / 12.9.10 |
| CUDA runtime / NVRTC包 | 均为12.6.77 |

系统需可用的Vulkan/RTX图形环境及libGLU.so.1。历史检查覆盖AppLauncher、单布局X5场景/三路相机和一次EvalEnv到假策略的动作；不代表真实ACT完整闭环。每次使用GPU前须重新核对UUID、显存和利用率；不以历史空闲快照选择设备。

Isaac Lab来源必须固定到RoboDojo配套fork提交afca7b09d60d8beb9c1cb28b43066499940b969b，并应用本目录的isaaclab-starlette.patch。约束文件constraints-isaac51.txt记录当前平台的实测版本，不是完整可移植锁文件，未做空白机器端到端复现。补丁原因、应用方式及验证边界见[兼容说明](../../docs/pusht/ISAACLAB_COMPATIBILITY.md)。

五个Lab包使用基础依赖，未启用强化学习或Robomimic extras。RoboDojo/XPolicyLab源码与设备补丁要求见[设备与客户端说明](../../docs/pusht/DEVICE_AND_CLIENT.md)；资产和规划要求见[资产与规划检查](../../docs/pusht/ASSETS_AND_PLANNER.md)。CuRobo追加约束见constraints-curobo-cu12.txt。依赖和资产并非完整可移植锁定环境。

数据接口使用h5py3.16.0、huggingface_hub0.36.2、NumPy1.26.0、OpenCV和Pillow；官方图像解码要求XPolicyLab源码可导入。仿真/训练环境已完成真实ACT单批次、短训练和模型重载的GPU检查；实际范围见[数据说明](../../docs/pusht/DATA_AND_ACT.md)。两条示范的PyTorch Dataset仅在CPU验收，不能据此声称ACT训练或推理支持CPU。

独立服务端已验证组合为Python3.10.22、torch2.4.1+cu121、torchvision0.19.1+cu121、NumPy1.26.4与websockets16.1.1，支持当前诊断checkpoint加载和单请求GPU推理。模型服务须显式禁用cudnn/matmul TF32，独立设备隔离与运行条件见[单动作说明](../../docs/pusht/ACT_EVAL_LINK.md)；仿真端websockets12.0可作为客户端，但缺少服务端依赖的websockets.asyncio。不要为运行服务端无目的升级Isaac依赖。

补丁robodojo-planner-device.patch用于对齐渲染、CUDA规划与相机输出缓冲区设备，同时保持原有CPU物理配置。新增客户端版本记录在constraints-client.txt。整合验证的范围与设备要求见[设备与客户端说明](../../docs/pusht/DEVICE_AND_CLIENT.md)；每次GPU运行前均须重新检查实时空闲，不能使用历史编号快照代替。
