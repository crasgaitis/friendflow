const flags = ['🌏', '🌎', '🌍'];
const flagContainer = document.querySelector('#flag-container');

function getRandomNumber(max) {
  return Math.floor(Math.random() * max);
}

function createFlag() {
  const flag = document.createElement('span');
  flag.classList.add('flag');
  flag.textContent = flags[getRandomNumber(flags.length)];
  flag.style.left = `${getRandomNumber(window.innerWidth)}px`;
  flag.style.top = `${getRandomNumber(window.innerHeight)}px`;
  return flag;
}

function spawnFlags() {
  setInterval(() => {
    const flag = createFlag();
    flagContainer.appendChild(flag);
    setTimeout(() => {
    flag.remove();
    }, 1500);
  }, 500);
}

spawnFlags();