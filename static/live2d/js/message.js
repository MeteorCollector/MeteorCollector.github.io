
function init(){
    var resourcesPaths = `${resourcesPath}`;
    var backImageNames = `${backImageName}`;
    var modelDirString = `${modelDir}`;
    var modelDirs = modelDirString.split(',');

    initDefine(resourcesPaths, backImageNames, modelDirs);
}

// 监听复制
(function() {
    document.addEventListener('copy',(e)=>{
      e.preventDefault();
      e.stopPropagation();
      showMessage('iris看到你在复制了（盯）', 5000, true)
    })
  }());

  $('.tool .fui-home').click(function (){
    //window.location = 'https://www.fghrsh.net/';
    window.location = window.location.protocol+'//'+window.location.hostname+'/'
});

$('.tool .fui-chat').click(function (){
    var items = ['其实，iris并不像你这样累','感觉不如去qq里说话...','你好呀！','在这里的只是我的投影罢了（笑），iris是没有实体的凹','我好快乐，因为我是iris！','(    = w =  )','人类为什么对随机这么着迷呢......','可以去听听《我们的田野》','人类的爱是什么东西呢？','您太重了！','你情就是命令，防你就是责任！','笑了，找iris真身找到这来了','在特定角度下，模型可能会崩坏。别被吓到（笑）','你可能注意到视线跟踪有一些漂移...容忍下这些bug吧，毕竟作者不会写网页...','你可能注意到视线跟踪有一些漂移...容忍下这些bug吧，毕竟作者不会写网页...','假如我可以知道你在想些什么，那么我将尝试减轻你的痛苦','太阳，我在这里！','太阳，你在哪里？','iris是谁？你？','摸了','在无意义的事情上浪费时间，可能这就是...咳，抱歉，接下来忘了。','感觉不如......到qq上和我说话','还是要多注意休息','诶？','在呢','iris现在是清醒状态（笑）'];
    text = items[Math.floor(Math.random()*items.length)];
    showMessage(text, 5000);
});

$('.tool .fui-user').click(function (){
    text = '这个按钮暂时没有用...';
    showMessage(text, 5000);
});

$('.tool .fui-info-circle').click(function (){
    //window.open('https://imjad.cn/archives/lab/add-dynamic-poster-girl-with-live2d-to-your-blog-02');
    window.open('https://meteorcollector.github.io/2022/10/iris-manual/');
});

$('.tool .fui-cross').click(function (){
    sessionStorage.setItem('waifu-dsiplay', 'none');
    showMessage('这页面上就我一个，你还要把我关了？', 5000, true);
});

$('.tool .fui-photo').click(function (){
    showMessage('这里不run许拍照。', 5000, true);
    window.Live2D.captureName = 'Pio.png';
    window.Live2D.captureFrame = true;
});

(function (){
    var text;
    //var SiteIndexUrl = 'https://www.fghrsh.net/';  // 手动指定主页
    var SiteIndexUrl = window.location.protocol+'//'+window.location.hostname+'/';  // 自动获取主页
    
    //if (window.location.href == SiteIndexUrl) {      // 如果是主页
    if (true) {
        var now = (new Date()).getHours();
        if (now > 23 || now <= 5) {
            text = 'iris现在是清醒状态...但是这个时间对于人类来说还是太晚了，快去休息吧~';
        } else if (now > 5 && now <= 7) {
            text = '早上好~iris今天也在轨运行中~';
        } else if (now > 7 && now <= 11) {
            text = '上午好~iris今天也在轨运行中~';
        } else if (now > 11 && now <= 14) {
            text = '中午好~（打哈欠）该昏睡一下了~';
        } else if (now > 14 && now <= 17) {
            text = '下午好~iris今天也在轨运行中~';
        } else if (now > 17 && now <= 19) {
            text = '现在可能是iris最喜欢的黄昏时刻...不知道有没有晚霞呢？';
        } else if (now > 19 && now <= 21) {
            text = '晚上好~iris今天也在轨运行中~';
        } else if (now > 21 && now <= 23) {
            text = '';
        } else {
            text = '其实，iris并不像你这么累';
        }
    } else {
        if(document.referrer !== ''){
            var referrer = document.createElement('a');
            referrer.href = document.referrer;
            var domain = referrer.hostname.split('.')[1];
            if (window.location.hostname == referrer.hostname) {
                text = '欢迎来到<span style="color:#0099cc;">"' + document.title.split(' - ')[0] + '"......</span>';
            } else if (domain == 'baidu') {
                text = '你好! 来自 百度搜索 的朋友<br>你是搜索 <span style="color:#0099cc;">' + referrer.search.split('&wd=')[1].split('&')[0] + '</span> 找到的我吗？';
            } else if (domain == 'so') {
                text = '你好! 来自 360搜索 的朋友<br>你是搜索 <span style="color:#0099cc;">' + referrer.search.split('&q=')[1].split('&')[0] + '</span> 找到的我吗？';
            } else if (domain == 'google') {
                text = '你好! 来自 谷歌搜索 的朋友<br>欢迎阅读<span style="color:#0099cc;">『' + document.title.split(' - ')[0] + '』</span>';
            } else {
                text = '你好! 来自 <span style="color:#0099cc;">' + referrer.hostname + '</span> 的朋友';
            }
        } else {
            text = '欢迎阅读<span style="color:#0099cc;">『' + document.title.split(' - ')[0] + '』</span>';
        }
    }
    showMessage(text, 6000);
})();

//window.hitokotoTimer = window.setInterval(showHitokoto,30000);
/* 检测用户活动状态，并在空闲时 定时显示一言 */
var getActed = false;
window.hitokotoTimer = 0;
var hitokotoInterval = false;

$(document).mousemove(function(e){getActed = true;}).keydown(function(){getActed = true;});
setInterval(function() { if (!getActed) ifActed(); else elseActed(); }, 1000);

function ifActed() {
    if (!hitokotoInterval) {
        hitokotoInterval = true;
        //hitokotoTimer = window.setInterval(showHitokoto, 30000);
    }
}

function elseActed() {
    getActed = hitokotoInterval = false;
    window.clearInterval(hitokotoTimer);
}

function showHitokoto(){
	/* 增加 hitokoto.cn API */
    text = '感觉不如...去qq上和我说话';
    showMessage(text, 5000);
	/*
	$.getJSON('https://api.fghrsh.net/hitokoto/rand/?encode=jsc&uid=3335',function(result){
        var text = '这句一言出处是 <span style="color:#0099cc;">『{source}』</span>，是 <span style="color:#0099cc;">FGHRSH</span> 在 {date} 收藏的！';
        text = text.render({source: result.source, date: result.date});
        showMessage(result.hitokoto, 5000);
        window.setTimeout(function() {showMessage(text, 3000);}, 5000);
    });
	*/
}

function showMessage(text, timeout, flag){
    if(flag || sessionStorage.getItem('waifu-text') === '' || sessionStorage.getItem('waifu-text') === null){
        if(Array.isArray(text)) text = text[Math.floor(Math.random() * text.length + 1)-1];
        //console.log(text);
        if(flag) sessionStorage.setItem('waifu-text', text);
        $('.live2d-tips').stop();
        $('.live2d-tips').html(text).fadeTo(200, 1);
        if (timeout === undefined) timeout = 5000;
        hideMessage(timeout);
    }
}

function hideMessage(timeout){
    $('.live2d-tips').stop().css('opacity',1);
    if (timeout === undefined) timeout = 5000;
    window.setTimeout(function() {sessionStorage.removeItem('waifu-text')}, timeout);
    $('.live2d-tips').delay(timeout).fadeTo(200, 0);
}