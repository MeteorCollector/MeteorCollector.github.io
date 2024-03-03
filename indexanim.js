function getRandomColor() {
  const colors = ['cyan', 'yellow', 'lime', 'blue', 'pink', 'magenta', 'orange'];
  const randomIndex = Math.floor(Math.random() * colors.length);
  return colors[randomIndex];
}

function createCircle() {
  const circle = document.createElement('div');
  circle.classList.add('circle');
  
  const size = Math.random() * 1000 + 200;
  const left = Math.random() * (window.innerWidth / 2) + window.innerWidth / 2;
  const color = getRandomColor();
  const animationDuration = Math.random() * 10 + 5;
  const delay = Math.random() * 5;
  
  circle.style.width = `${size}px`;
  circle.style.height = `${size}px`;
  circle.style.left = `${left}px`;
  circle.style.background = color;
  
  document.querySelector('.circles-container').appendChild(circle);
  
  circle.addEventListener('animationend', () => {
    circle.remove();
  });

  setTimeout(() => {
    circle.classList.add('animate');
  }, 10);
}

setInterval(createCircle, 2000);
