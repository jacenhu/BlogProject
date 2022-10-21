# linux系统命令

## yum

| 语法 | 作用 |
| :---: | :---: |
| yum list installed | 显式已经安装的包 |
| yum list xxx | 查找可安装xx的包 |
| yum install xxx | 安装xxx的包 |
| yum remove xxx | 卸载xxx的包 |
| yum deplist xxx | 列出xxx的包依赖 |
| yum -y | 自动选择是 |
| yum info | 显式软件包的信息 |
| yum update | 升级所有的包 |
| yum check-update | 检查可更新的程序 |

## 更换centos镜像

wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo

grep -rl 'cloud.aliyuncs' ./ | xargs sed -i 's/cloud.aliyuncs/aliyun/g'

yum clean all

yum makecache

## 更换epel源

wget -O /etc/yum.repos.d/epel.repo http://mirrors.aliyun.com/repo/epel-7.repo

yum clean all

yum makecache

## gcc编译

yum -y install gmp-devel
yum -y install mpfr-devel
yum -y install libmpc-devel
yum -y install bzip2
wget https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz
./contrib/download_prerequisites
mkdir gcc-build-7.3.0
cd gcc-build-7.3.0
../configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc7.3.0
make -j
export LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:$LD_LIBRARY_PATH

## 创建交换分区

sudo dd if=/dev/zero of=/swapfile bs=1M count=4096    # 1 * 2048 = 2048 创建 1 g 的内存分区
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapoff /swapfile
sudo rm /swapfile

## 参考资料

CentOS 初体验三： Yum 安装、卸载软件
https://blog.csdn.net/zhaoyanjun6/article/details/78894974

Redhat 或者centos 更换第三方epel源
https://developer.aliyun.com/article/516685