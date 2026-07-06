/**
 * Neural Network Particle Background
 * Draws animated nodes + connecting lines on a canvas
 */
(function() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let particles = [];
  let animFrame;

  const CONFIG = {
    count: 40,          // reduced from 55 — cuts O(n²) line checks from 1485→780 per frame
    maxDist: 130,
    speed: 0.35,
    nodeRadius: 2,
    colors: {
      node: 'rgba(0, 212, 255, 0.7)',
      nodeAlt: 'rgba(124, 58, 237, 0.6)',
      line: 'rgba(0, 212, 255, '
    }
  };

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function createParticle() {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * CONFIG.speed,
      vy: (Math.random() - 0.5) * CONFIG.speed,
      r: Math.random() * 1.5 + CONFIG.nodeRadius,
      isAlt: Math.random() > 0.7
    };
  }

  function init() {
    particles = Array.from({ length: CONFIG.count }, createParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONFIG.maxDist) {
          const opacity = (1 - dist / CONFIG.maxDist) * 0.35;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = CONFIG.colors.line + opacity + ')';
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.isAlt ? CONFIG.colors.nodeAlt : CONFIG.colors.node;
      ctx.shadowBlur = 8;
      ctx.shadowColor = p.isAlt ? '#7c3aed' : '#00d4ff';
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }

  function update() {
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });
  }

  function loop() {
    update();
    draw();
    animFrame = requestAnimationFrame(loop);
  }

  window.addEventListener('resize', () => { resize(); init(); });
  resize();
  init();
  loop();
})();
