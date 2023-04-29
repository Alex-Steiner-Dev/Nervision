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

  const ambientLight = new THREE.AmbientLight(0xffffff, 1)
  scene.add(ambientLight)

  const mtlLoader = new THREE.MTLLoader();

  mtlLoader.load(modelUrl.replace(".obj", ".mtl"), function(materials) {
    materials.preload();
  
    const objLoader = new THREE.OBJLoader();

    objLoader.setMaterials(materials);

    objLoader.load(modelUrl, function(object) {
      scene.add(object);
    });
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