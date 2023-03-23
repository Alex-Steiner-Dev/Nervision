// Initialize
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

const fov = 45;
const aspect = canvas.width / canvas.height;
const near = 0.1;
const far = 100;


// Load Model
function loadModel(modelUrl){
  const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);

  camera.position.set(0, 0, 15); 
  camera.lookAt(new THREE.Vector3(0, 0, 0));

  const controls = new THREE.OrbitControls(camera, canvas);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#FFFFFF');

  const ambientLight = new THREE.AmbientLight('#fff', 1);
  scene.add(ambientLight);

  const loader = new THREE.OBJLoader();

  loader.load(modelUrl, (obj) => {
    const model = obj;

    model.scale.set(7, 7, 7);
    model.position.set(0, -1, 0);

    const material = new THREE.MeshBasicMaterial({ color: '#000000' });
    model.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material.color = material.color;
      }
    });

    scene.add(model);
  });

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio * 2); 

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    controls.update();
  }

  animate();
}

loadModel("/static/generation.obj");