# 使用vuePress搭建个人博客

[[toc]]

## 一、安装node.js
1. 进入[node.js下载页面](https://nodejs.org/en/download/)，根据操作系统类型选择特定版本下载。
![node-download-pic](/img/vuepress/node-download.png)
2. MacOS下，点击node-v12.pkg直接安装，各自安装至以下目录
* Node.js v12.14.1 to /usr/local/bin/node
* npm v6.13.4 to /usr/local/bin/npm
3. 安装完成后，在terminal下，使用以下命令查看版本号
* node -v
* npm -v
![node-version-pic](/img/vuepress/node-version.png)

## 二、安装vuePress,搭建博客框架
1. 命令行输入：npm install -g vuepress
全局安装vuepress
2. 搭建博客框架

2.1 创建博客目录 mkdir myBlog

2.2 创建子目录
* /docs
* /docs/.vuepress
* /docs/

2.3 创建js文件
* config.js
* nav.js
* sidebar.js

2.4 然后按照[vuepress官方文档](https://vuepress.vuejs.org/zh/guide/)编写各个文件的内容。

2.5 本地调试
* 构建静态文件 npm run build
* 发布 npm run dev
出现如下图片，即编译成功。
![npm-run-dev-pic](/img/vuepress/npm-run-dev.png)
成功后，在浏览器窗口，输入 http://localhost:8080/ 访问本地网站
![localhost-site-pic](/img/vuepress/localhost-site.png)

## 三、安装nginx
> 本文档的开发环境：

类别|环境|备注
:--:|:--:|:--:
服务器|弹性云ECS|
操作系统|CentOS7|
ssh工具|Royal TSX|
1. 添加yum源 
sudo rpm -ivh http://nginx.org/packages/centos/7/noarch/RPMS/nginx-release-centos-7-0.el7.ngx.noarch.rpm
2. 安装nginx  
sudo yum install nginx
3. 查看nginx安装路径  
rpm -ql nginx
![nginx-config-pic](/img/vuepress/nginx-config.png)
4. 本地编译dist包，scp到服务器上
scp -r dist root@47.107.82.83:/opt/www/blog
5. 编辑nginx配置文件
首先注释掉这一行
```
# include /etc/nginx/conf.d/*.conf;
```
然后，编写server部分
```
server {
      listen  80;
      server_name  jacenhu.site;
      root /opt/www/blog/dist;

      location / {
        #root html
        root  /opt/www/blog/dist;
        index  index.html index.htm;
      }
    }
```
最后，检查配置文件是否正确 
执行，nginx -t ，若出现以下两句，则语法无错误
```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successfu
```
6. 一些nginx的常用命令

6.1 卸载nginx rpm -e nginx

6.2 服务启动/停止
* 设置开机启动
  sudo systemctl enable nginx
* 启动服务
  sudo systemctl start nginx
* 重启服务
  sudo systemctl restart nginx
  nginx -s reload
  
7. 配置完成后，地址栏输入: http://ip，即可访问
如： http://X.X.X.X
![nginx-ip-pic](/img/vuepress/nginx-ip.png)

## 四、域名解析
> 每次通过http://ip方式登录，需要输入很多数字，比较麻烦。因此申请一个简易好记的域名十分重要。
1. 此前已经申请好了域名 www.jancehu.site，通过云DNS解析，将其映射到X.X.X.X
* 注：域名解析是指把域名指向网站空间IP
2. 访问 [www.jacenhu.site](http://www.jacenhu.site)

## 五、开发小知识点


