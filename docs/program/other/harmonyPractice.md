# Hello world 鸿蒙OS App小实践

今天华为开发者大会发布了Harmony OS 2.0, 作为一个搬砖工，自然也不能落后，也来玩一玩鸿蒙系统。


## 鸿蒙OS网站
首先进入进入开发者联盟网站，找到鸿蒙OS。

https://developer.huawei.com/consumer/cn/

![hm_devSite](/img/hm/hm_devSite.png)

点击进一步了解，[鸿蒙OS网站](https://www.harmonyos.com/cn/home)

## 小试牛刀


（1）注册华为开发者账号

![hm_devAccount](/img/hm/hm_devAccount.png)

（2）下载DevEco Studio

网址：https://developer.harmonyos.com/cn/develop/deveco-studio

点击立即下载：

![hm_devStudio](/img/hm/hm_devStudio.png)



![hm_devStudioDownload](/img/hm/hm_devStudioDownload.png)

一共616MB，还是有点大的。

（3）安装IDE

点击exe进行安装，一路next。

![hm_devStudioInstall1](/img/hm/hm_devStudioInstall1.png)

![hm_devStudioInstall2](/img/hm/hm_devStudioInstall2.png)

![hm_devStudioInstall3](/img/hm/hm_devStudioInstall3.png)





（4）配置开发环境

首先，打开IDE的时候，需要下载toolchain和java sdk
![hm_devStudioInstall4](/img/hm/hm_devStudioInstall4.png)

![hm_devStudioInstall5](/img/hm/hm_devStudioInstall5.png)

下载SDK。

Configure -> Settings -> Appearance & Behavior ->  System Settings -> HarmonyOS SDK -> SDK(API Version 3)

![hm_devStudioInstall6](/img/hm/hm_devStudioInstall6.png)

点击Apply就会开始下载了

![hm_devStudioInstall7](/img/hm/hm_devStudioInstall7.png)

（5）第一个Harmony OS 程序： **Hello World**

创建新工程，选择ability template，按照官网，我选择的是Wearable。

![hm_devStudioInstall8](/img/hm/hm_devStudioInstall8.png)

![hm_devStudioInstall9](/img/hm/hm_devStudioInstall9.png)

代码目录如下：

![hm_devStudioInstall10](/img/hm/hm_devStudioInstall10.png)

废了九牛二虎之力，模拟器终于下载下来。然后运行程序。
点击Toos->HVD Manager，会转到华为开发者联盟网站。

![hm_devStudioInstall12](/img/hm/hm_devStudioInstall12.png)

![hm_devStudioInstall11](/img/hm/hm_devStudioInstall11.png)

点击允许，会提示已经成功登录客户端。

![hm_devStudioInstall13](/img/hm/hm_devStudioInstall13.png)

IDE页面点击Agree

![hm_devStudioInstall14](/img/hm/hm_devStudioInstall14.png)

然后会弹出Device Manager，Actions中运行。

![hm_devStudioInstall15](/img/hm/hm_devStudioInstall15.png)

弹出模拟器。

![hm_devStudioInstall16](/img/hm/hm_devStudioInstall16.png)

Run Entry运行工程，选择Target。

![hm_devStudioInstall17](/img/hm/hm_devStudioInstall17.png)

点击OK，运行完成后，手表就出现了Hello World。

![hm_devStudioInstall18](/img/hm/hm_devStudioInstall18.png)










