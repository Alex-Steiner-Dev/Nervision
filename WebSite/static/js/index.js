const canvas = document.getElementById('model-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

const fov = 45;
const aspect = canvas.width / canvas.height;
const near = 0.1;
const far = 100;
const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
camera.position.set(0, 0, 15); 
camera.lookAt(new THREE.Vector3(0, 0, 0));

const controls = new THREE.OrbitControls(camera, canvas);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#111');

const ambientLight = new THREE.AmbientLight('#fff', 1);
scene.add(ambientLight);

const loader = new THREE.GLTFLoader();
const modelUrl = '/static/chair.glb';

loader.load(modelUrl, (gltf) => {
  const model = gltf.scene;
  model.position.set(0, 2, 0)
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