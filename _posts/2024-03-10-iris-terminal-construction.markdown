---
layout: post
title:  "iris直播迁移笔记（上）"
date:   2024-03-10 14:42:00 +0800
categories: posts
tag: iris
---

## 前情提要

iris在B站播了已经有一个月了，但是由于iris直播间真的没有什么吸引力，每天只是挂着而已。再加上b站的诸多限制，我感觉意义已经不大。所以我产生了把直播间迁移到自己网站上的想法，说不定还可以实现更好的界面和更多的功能。

首先在信息收发方面，可以实现更好的前后端结合。另外，我觉得我可以试着把音乐广播转接到网页上来，这样iris的直播间就有了声音。

## 音乐广播

为了播音乐，我在滚石500张专辑里严选了几十张并把它们用 `mp3Wrap` 首尾相接制作了一个巨大的mp3音频，然后在云服务器上用rtmp推流。我的 `VLC Player ` 和移动端的 `km player ` 都可以很好地收到推流。但是我在把它与视频画面合流的时候总会出现问题。我推测是因为时间戳。

### 配置hls

自然地，我开始寻找把rtmp接到web page上的方法。但是结果却是巨大的"NO"，因为rtmp需要flash才能在网页上播放，然而flash早就寿终正寝了。

于是我把目光转向其他通讯协议，发现了这个网页：

[https://stackoverflow.com/questions/19658216/how-can-we-transcode-live-rtmp-stream-to-live-hls-stream-using-ffmpeg](https://stackoverflow.com/questions/19658216/how-can-we-transcode-live-rtmp-stream-to-live-hls-stream-using-ffmpeg)

上面介绍了rtmp转hls的方法。

于是我按照上面的配置方法建立了hls推流，非常顺利，`VLC Player ` 和 `km player ` 都可以听到推流。自然地，我搞来了一段hls播放器代码：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>iris live</title>
    <link href="https://vjs.zencdn.net/7.4.1/video-js.css" rel="stylesheet">
    <script src='https://vjs.zencdn.net/7.4.1/video.js'></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/videojs-contrib-hls/5.15.0/videojs-contrib-hls.min.js" type="text/javascript"></script>
</head>
<body>
    <div class="radiocard">
        <style>
            .video-js .vjs-tech {position: relative !important;}
        </style>
        <div>
            <video id="myVideo" class="video-js vjs-default-skin vjs-big-play-centered" controls preload="auto" data-setup='{}' style='width: 100%;height: auto'>
                <source id="source" src="http://host:port/name.m3u8" type="application/x-mpegURL"></source>
            </video>
        </div>
    </div>
</body>
```

不出所料地没有播放成功。这时候我想，是不是因为我推流的是音频，所以不能在video标签里播放呢？

所以我开始着手把音频信号转成视频信号。很简单，加个画面就可以了。于是我用 `ffmpeg` 给音频加上一个 $4 \times 4$ 的纯蓝色画面，发现视频大小超出了最大限制。于是先生成了一个10秒的蓝色视频：

```bash
ffmpeg -f lavfi -i color=c=blue:s=4x4 -t 10 blue_screen.mp4
```

然后循环推流这个蓝色视频，并且和音频合流：

```bash
ffmpeg -fflags +genpts -stream_loop -1 -i blue_screen.mp4 -i audio.mp3 -shortest -map 0:v -map 1:a -c:v copy -c:a aac -f flv rtmp://localhost:port/hls/mystream
```

我之前尝试了 `-c:a copy`，发现播不出声音，做一次转码就可以了。正常来讲是可以直接用copy可以播出来的，而且可以节省一些资源。我认为是音频文件过长，拼接过程中带来了一些问题。读者不妨尝试 `-c:a copy`。

### 规则疑云

视频调试出来之后，我的 `VLC Player ` 和移动端的 `km player ` 又都可以很好地收到蓝色推流。这时候我开始找在线的hls测试网站，如果在网页上测试到我的推流没问题，那我就可以成功地把视频流推到自己的网站上了。

推荐测试网站：[livepush.io](https://livepush.io/hls-player/index.html)    [hlsjs.video-dev.org](https://hlsjs.video-dev.org/demo/)    [hlsplayer.org](https://hlsplayer.org/)

果然是报错了，报错内容为：

```
Cannot load http://(...).m3u8 HTTP response code:0 This might be a CORS issue, consider installing Allow-Control-Allow-Origin Chrome Extension
```

出现了CORS issue，[也就是跨域资源共享(Cross-Origin Resource Sharing)问题](https://blog.csdn.net/yujia_666/article/details/108490178)。由于前后端不在一个域，所以加载资源的时候会出现问题。这个时候需要配置一下`nginx`，核心在于要对目标域名打开 `Access-Control-Allow-Origin` 。我的配置文件在 `/etc/nginx/nginx.config`，在 `hls` 应用处设置：

```
# Client (VLC etc.) can access HLS here.
location /hls {
    # Serve HLS fragments
    types {
        application/vnd.apple.mpegurl m3u8;
        video/mp2t ts;
    }
    root /tmp;
    add_header Cache-Control no-cache;
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Expose-Headers' 'Content-Length'  always;
    add_header 'Access-Control-Allow-Headers' 'Origin,Range,Accept-Encoding,Referer,Cache-Control'  always;
    # add_header 'Access-Control-Expose-Headers' 'Server,Content-Length,Content-Range,Date';
    add_header 'Access-Control-Allow-Methods' 'GET, HEAD, POST, OPTIONS, PUT, DELETE' always;
}
```

也可以全局设置，配置方法在[这里](https://enable-cors.org/server_nginx.html)。

值得一提的是，在[COR检查工具](https://cors-test.codehappy.dev/)中，无论如何配置都说我的服务器 `does not work properly with CORS`，但是直接发起请求（可以用这个[在线工具](https://reqbin.com/post-online)）的话可以发现header是完全正确的：

```
HTTP/1.1 200 OK
Server: nginx/1.18.0 (Ubuntu)
Date: Sat, 09 Mar 2024 06:24:08 GMT
Content-Type: applicationnd.apple.mpegurl
Content-Length: 268
Last-Modified: Sat, 09 Mar 2024 06:24:07 GMT
Connection: keep-alive
ETag: "65ec0087-10c"
Cache-Control: no-cache
Access-Control-Allow-Origin: *
Accept-Ranges: bytes
```

这时候我一头雾水，于是打开了本地的html（就是上面那个播放hls的html文件），发现音频被完美地播放了出来！

wtf？

之后我把这个页面放到了 `gitpage` 的某个目录，发现 `edge` 和 `chrome` 上又都播放不出来，手机浏览器上却可以。

这时候我终于想起来可以按 `F12` 看看 `console` 的报错信息。果不其然，看了之后就知道怎么回事了。

 <p><img src="{{site.url}}/images/webconsole.png" width="60%" align="middle" /></p>

原来是浏览器拒绝了不安全的 `http` 请求......

这时候我有两个方案，一个是把服务端升到 `https` ，一种是把前端降到 `http`。不过我并没有钱和精力搞证书，所以我只有采用后面一种方案了。现在这个网页被托管到我自己的服务器，[通过ip地址直接访问](http://120.46.209.170/livestream.html)。 

### 回到原点

这时候我又想到，是不是一开始就是CORS和http的问题，和音视频没有关系呢？于是我把推流指令改回成了最开始的音频指令

```bash
ffmpeg -fflags +genpts -re -stream_loop -1 -i music.mp3 -vn -c:a aac -ar 44100 -b:a 128k -f flv rtmp://host:port/streamname
```

果然音频是可以被播放的。但是这时候服务端又出了问题，每播放一段时间之后就会出现时间戳报错，例如

```
[flv @ 0x55fa051841c0] Non-monotonous DTS in output stream 0:0; previous: 14723792, current: 10203677; changing to 14723792. This may result in incorrect timestamps in the output file.
```

这确实是时间戳出了问题。所以我弃用了那个几十张专辑拼接而成的巨大 `mp3`，改成了文件夹下专辑原本 `mp3` 文件的轮播。编写 `bash` 脚本：

```bash
#!/bin/bash
input_folder="/path/to/folder"
stream_url="rtmp://host:port/streamname"

while true; do
    for file in "$input_folder"/*.mp3; do
    filename=$(basename -- "$file")
    filename="${filename%.*}"

    ffmpeg -re -i "$file" -c:a aac -ar 44100 -b:a 128k -f flv "$stream_url"
    done
done
```

至此，关于音频直播的一切才配置完毕。