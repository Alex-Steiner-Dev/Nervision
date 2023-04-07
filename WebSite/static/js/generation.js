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

  // create an ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  // create a directional light
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight.position.set(0, 1, 0);
  scene.add(directionalLight);

  const loader = new THREE.GLTFLoader();

  loader.load(modelUrl, (gltf) => {
    gltf.encoding = THREE.sRGBEncoding;
    const model = gltf.scene;
    const texture = gltf.textures;
    
    model.traverse((child) => {
      if (child.isMesh) {
        child.material.map = texture;
      }
    });

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

loadModel("/static/1.glb");