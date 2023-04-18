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

  const ambient = new THREE.AmbientLight(0x404040, 5)
  const point = new THREE.PointLight(0xE4FF00, 1, 10)
  point.position.set(3, 3, 2)
  scene.add(ambient)
  scene.add(point)

  const loader = new THREE.GLTFLoader();

  loader.load(modelUrl, (gltf) => {
    gltf.encoding = THREE.sRGBEncoding;
    const model = gltf.scene;
   

    model.scale.set(7, 7, 7);
    model.position.set(0, -1, 0);

   
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

loadModel("/static/models/model1.glb");