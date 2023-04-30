// Initialize
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

// Load Model
function loadModel(modelUrl){
  camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.1, 100 );
  camera.position.set( 1.5, 4, 9 );

  scene = new THREE.Scene();
  scene.background = new THREE.Color( 0xffffff );

  // Load Light
  var ambientLight = new THREE.AmbientLight( 0xcccccc );
  scene.add( ambientLight );
        
  var directionalLight = new THREE.DirectionalLight( 0xffffff );
  directionalLight.position.set( 0, 1, 1 ).normalize();
  scene.add( directionalLight );				


  const loader = new THREE.GLTFLoader();

  loader.load(modelUrl , function ( gltf ) {

    scene.add( gltf.scene );

    render();

  } );

  renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize( window.innerWidth, window.innerHeight );
  document.body.appendChild( renderer.domElement );

  const controls = new THREE.OrbitControls( camera, renderer.domElement );
  controls.addEventListener( 'change', render );
  controls.target.set( 0, 2, 0 );
  controls.update();

  window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize( window.innerWidth, window.innerHeight );

  render();

}

function render() {

  renderer.render( scene, camera );

}