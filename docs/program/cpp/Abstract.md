# 计算机术语概念解释（持续更新）
## 大数据领域
* OLTP: On-Line Transaction Processing 联机事务处理过程。其基本特征是前台接收的用户数据可以立即传送到计算中心进行处理，并在很短的时间内给出处理结果，是对用户操作快速响应的方式之一。
* OLAP: On-Line Analytical Processing 联机分析处理。OLAP是依赖于OLTP的，因为OLAP分析的数据都是由OLTP所产生的，也可以看作OLAP是OLTP的一种延展，一个让OLTP产生的数据发现价值的过程。
## 云存储领域
* RDMA: Remote Direct Memory Access, 全称远程直接内存访问。为了解决网络传输中客户端与服务器端数据处理的延迟而产生的。它将数据直接从一台计算机的内存传输到另一台计算机，无需双方操作系统的介入。这允许高吞吐、低延迟的网络通信，尤其适合在大规模并行计算机集群中使用。
> 计算机网络通信中最重要的两个衡量指标主要是指高带宽和低延迟。

> Remote：数据通过网络与远程机器间进行数据传输。
Direct：没有内核的参与，有关发送传输的所有内容都卸载到网卡上。
Memory：在用户空间虚拟内存与RNIC网卡直接进行数据传输不涉及到系统内核，没有额外的数据移动和复制。
Access：send、receive、read、write、atomic操作。

## 参考资料
* [常见的几个大数据名词：OLAP、OLTP、BI到底代表着什么？](https://blog.csdn.net/qq_41946557/article/details/103171150)
* [OLTP与OLAP的关系是什么？](https://www.zhihu.com/question/24110442/answer/851671343)
* [深入浅出全面解析RDMA](https://blog.csdn.net/qq_21125183/article/details/80563463)




