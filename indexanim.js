function createCircle() {
  const circle = document.createElement('div');
  circle.classList.add('circle');
  
  // 随机设置圆的位置、大小和动画延迟时间
  const size = Math.random() * 100 + 50; // 随机大小
  const left = Math.random() * window.innerWidth; // 随机水平位置
  const animationDuration = Math.random() * 10 + 5; // 随机动画持续时间
  const delay = Math.random() * 5; // 随机延迟时间
  
  circle.style.width = `${size}px`;
  circle.style.height = `${size}px`;
  circle.style.left = `${left}px`;
  circle.style.animationDuration = `${animationDuration}s`;
  circle.style.animationDelay = `${delay}s`;
  
  // 将圆添加到页面中
  document.querySelector('.circles-container').appendChild(circle);
  
  // 在动画结束后移除圆
  circle.addEventListener('animationend', () => {
    circle.remove();
  });
}

// 创建圆的定时器
setInterval(createCircle, 2000);
