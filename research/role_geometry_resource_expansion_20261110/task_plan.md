# GPU0与GPU3各新增一个常驻worker

用户最新授权：在当前六worker基础上，GPU0与GPU3各新增一个，总上限8；GPU1原4、GPU0变3、GPU3新增1。复用当前1440键队列和科学worker，不重载原六模型。

- [x] 读取当前入口、原管理与收尾逻辑；GPU资源实时只读核对。
- [x] 新增独立管理版本及CPU防护检查，锁定管理文件。
- [x] 核验原manager身份后仅替换管理进程，接管原六worker/MPS并依次新增两个worker。
- [x] 核验模型ready、原模型身份不变、唯一队列及科学SHA；记录两地并本地Git。
