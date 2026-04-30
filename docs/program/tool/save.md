# 收藏

[[toc]]

## vs code 有用的插件

（1）Material Theme

主题

## markdown的工具

（1） Emoji表情


https://github.com/markdown-it/markdown-it-emoji/blob/master/lib/data/full.json

例如

``` markdown
:100:
```

:100:

(2) 目录

``` markdown
[[toc]]
```

(3) 自定义容器

``` markdown
::: tip
这是一个提示
:::

::: warning
这是一个警告
:::

::: danger
这是一个危险警告
:::

::: details
这是一个详情块，在 IE / Edge 中不生效
:::
```

::: tip
这是一个提示
:::

::: warning
这是一个警告
:::

::: danger
这是一个危险警告
:::

::: detail
这是一个详情块，在 IE / Edge 中不生效
:::

::: details 点击查看代码

``` js
console.log('你好，VuePress！')
```

:::

details语法试验了，没有生效。可以使用H5标签。

``` markdown
<details>
    <summary>农夫与蛇</summary>
    ``` JS
    console.log(123456));
    ```
    一个农夫在寒冷的冬天里看见一条蛇冻僵了，觉得它很可怜，就把它拾起来，小心翼翼地揣进怀里，用暖热的身体温暖着它。那条蛇受到了暖气，渐渐复苏了，又恢复了生机。等到它彻底苏醒过来，便立即恢复了本性，用尖利的毒牙狠狠地咬了恩人一口，使他受到了致命的创伤。农夫临死的时候痛悔地说：“我可怜恶人，不辨好坏，结果害了自己，遭到这样的报应。 如果有来世 ，我绝不怜惜像毒蛇一样的恶人。”
</details>
```

<details>
    <summary>农夫与蛇</summary>
    一个农夫在寒冷的冬天里看见一条蛇冻僵了，觉得它很可怜，就把它拾起来，小心翼翼地揣进怀里，用暖热的身体温暖着它。那条蛇受到了暖气，渐渐复苏了，又恢复了生机。等到它彻底苏醒过来，便立即恢复了本性，用尖利的毒牙狠狠地咬了恩人一口，使他受到了致命的创伤。农夫临死的时候痛悔地说：“我可怜恶人，不辨好坏，结果害了自己，遭到这样的报应。 如果有来世 ，我绝不怜惜像毒蛇一样的恶人。”
</details>

## CI工具

* Travis CI

一个使用yaml格式配置用于持续集成完成自动化测试部署的开源项目

https://docs.travis-ci.com/user/deployment/pages/

## ECS异常
* 处理挖矿程序
https://help.aliyun.com/document_detail/161236.html?spm=5176.10695662.1996646101.searchclickresult.6739340dh4r4WY
