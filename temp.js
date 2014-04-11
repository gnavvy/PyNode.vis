// ***** call from server *****
now.setHands = function(hands) { _hands = hands; };
now.ready(function() { now.start(); });
now.update = function() { init(); animate(); };
// ***** call from server *****

var W, H, w, h;
var canvas, camera, scene, renderer, projector, stats;  // basic
var screen, floor;
var trackball, plane, fillip;
var _hands;
var objects = [];
var mouse = new THREE.Vector2();
var mousePrev = new THREE.Vector2();
var offset = new THREE.Vector3();
var v3 = new THREE.Vector3();
var INTERSECTED, SELECTED;
var print = console.log.bind(console);
var STATE = { NONE: 0, ROTATE: 1, SCALE: 2, TRANSLATE: 3 }, _state = STATE.NONE;

function init() {
    initCanvas();
    initScene();
    initLighting();
    // initRubix();
}

function initRubix() {
    var cube = rubix.createCube(0);
    cube.autoUpdateMatrix = false;
    cube.useQuaternion = true;
    cube.position.y = 200;
    scene.add(cube);
    objects.push(cube);
}

function initCanvas() {
    window.addEventListener('resize', onWindowResize, false);
    W = window.innerWidth;  H = window.innerHeight;
    w = 160; h = 90;

    // Canvas - full browser screen mode
    canvas = document.createElement("div");
    document.body.appendChild(canvas);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H, undefined);
    renderer.setFaceCulling(THREE.CullFaceNone, undefined);
    renderer.sortObjects = false;
    renderer.shadowMapEnabled = true;
    renderer.shadowMapType = THREE.PCFShadowMap;
    renderer.domElement.addEventListener('mousemove', onDocumentMouseMove, false);
    renderer.domElement.addEventListener('mousedown', onDocumentMouseDown, false);
    renderer.domElement.addEventListener('mouseup', onDocumentMouseUp, false);
    canvas.appendChild(renderer.domElement);

    // Stats
    stats = new Stats();
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.top = '0px';
    canvas.appendChild(stats.domElement);

    // Projector
    projector = new THREE.Projector();
}

function initScene() {
    // Scene
    scene = new THREE.Scene();

    // Camera
    camera = new THREE.PerspectiveCamera(45, W / H, 1, 1000000);
    camera.position.set(6000, 3000, 6000);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    scene.add(camera);

    // Trackball
    trackball = new THREE.TrackballControls(camera);
    trackball.staticMoving = true;
    trackball.addEventListener('change', render);

    // Axis
    scene.add(new THREE.AxisHelper(3200));

    // Coordinates
    planeMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(w*20, w*20, w/2, w/2),
        new THREE.MeshBasicMaterial({ color: 0xe0e0e0, wireframe: true })
    );

    xy_plane = planeMesh.clone();
    xz_plane = planeMesh.clone();
    yz_plane = planeMesh.clone();

    xy_plane.position.set(w*10, w*10, 0);
    xz_plane.rotation.x = Math.PI/2;
    xz_plane.position.set(w*10, 0, w*10);
    yz_plane.rotation.y = Math.PI/2;
    yz_plane.position.set(0, w*10, w*10);

    scene.add(xy_plane);
    scene.add(xz_plane);
    scene.add(yz_plane);
}

function initLighting() {
    var light1 = new THREE.DirectionalLight(0xffffff, 1); {
        light1.position.set(0, 500, 500);
        light1.castShadow = true;
        light1.shadowCameraNear = 200;
        light1.shadowCameraFar = camera.far;
        light1.shadowCameraFov = 50;
        light1.shadowBias = -0.00022;
        light1.shadowDarkness = 0.5;
        light1.shadowMapWidth = 2048;
        light1.shadowMapHeight = 2048;
    }
    var light2 = light1.clone(); {
        light2.intensity = 0.5;
        light2.position.set(-500, -500, -500);
    }
    var light3 = light1.clone(); {
        light3.intensity = 0.3;
        light3.position.set(500, -500, -500);
    }

    scene.add(light1);
    scene.add(light2);
    scene.add(light3);
}

function animate() {
    requestAnimationFrame(animate);
    render();
    stats.update();

    if (trackball.enabled) {
        trackball.update();
    }
}

function render() {
    if (_hands && _hands.length > 0) {
        objects[0].position.fromArray(_hands[0].stabilizedPalmPosition);
        objects[0].rotation.fromArray(_hands[0].direction);
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    W = window.innerWidth;  H = window.innerHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
}

function onDocumentMouseDown(event) {
    event.preventDefault();

    var vector = new THREE.Vector3(mouse.x, mouse.y, 0.5);
    projector.unprojectVector(vector, camera);

    var raycaster = new THREE.Raycaster(camera.position, vector.sub(camera.position).normalize());
    var candidates = raycaster.intersectObjects(objects, true);
    if (candidates.length > 0) {
        trackball.enabled = false;  // stop rotating camera
        SELECTED = candidates[0].object;  // first hit object
        while (!(SELECTED.parent instanceof THREE.Scene)) {  // select group
            SELECTED = SELECTED.parent;
        }
        _state = event.button === 0 ? STATE.ROTATE : STATE.TRANSLATE;
        if (_state === STATE.TRANSLATE) {
            offset.copy(raycaster.intersectObject(plane)[0].point).sub(plane.position);
            canvas.style.cursor = 'move';
        }
    }
}

function onDocumentMouseMove(event) {
    event.preventDefault();

    mousePrev.copy(mouse);

    mouse.x =  (event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    var vector = new THREE.Vector3(mouse.x, mouse.y, 0.5);
    projector.unprojectVector(vector, camera);

    if (_state === STATE.ROTATE) {
        var euler = new THREE.Vector2().subVectors(mouse, mousePrev).multiplyScalar(2);
        var targetQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Vector3(-euler.y, euler.x, 0));
        SELECTED.quaternion.multiplyQuaternions(SELECTED.quaternion, targetQuaternion);
        return;
    }

    var raycaster = new THREE.Raycaster(camera.position, vector.sub(camera.position).normalize());

    if (SELECTED) {  // an object is previously selected
        SELECTED.position.copy(raycaster.intersectObject(plane)[0].point.sub(offset));
        return;
    }

    var candidates = raycaster.intersectObjects(objects, true);
    if (candidates.length > 0) {
        if (INTERSECTED != candidates[0].object) {
            INTERSECTED = candidates[0].object;
            while (!(INTERSECTED.parent instanceof THREE.Scene)) {  // select group
                INTERSECTED = INTERSECTED.parent;
            }
            plane.position.copy(INTERSECTED.position);
            plane.lookAt(camera.position);  // face to the user
        }
        canvas.style.cursor = 'pointer';

    } else {
        INTERSECTED = null;
        canvas.style.cursor = 'auto';
    }
}

function onDocumentMouseUp(event) {
    event.preventDefault();
    _state = STATE.NONE;
    trackball.enabled = true;
    if (INTERSECTED) {
        plane.position.copy(INTERSECTED.position);
        SELECTED = null;
    }
    canvas.style.cursor = 'auto';
    print(camera.position)
}