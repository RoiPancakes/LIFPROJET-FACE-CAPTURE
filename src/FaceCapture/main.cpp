#include "window.h"

#include "mesh.h"
#include "wavefront.h" // pour charger un objet au format .obj
#include "orbiter.h"
#include "texture.h"

#include "draw.h" // pour dessiner du point de vue d'une camera
#include <stdio.h>


/* OpenCV et DLIB */
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>


using namespace dlib;
using namespace std;

// Image situant sur le visage les landmark retournés par dlib : https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg
// Attention les landmarks vont de 0 à 67 (et non de 1 à 68 comme présenté dans le shema)

/* Définition des variables servant au parcours dans les tableaux */
enum models{BASE, MOUTH_OPEN, DUCKFACE, ANGRY, EYEBROW_DOWN, EYEBROW_UP, EYE_CLOSED, NOSE_UP, SMILING, BLEND, SIZE};

double clamp(double x, double lowerlimit, double upperlimit) {
  if (x < lowerlimit)
    x = lowerlimit;
  if (x > upperlimit)
    x = upperlimit;
  return x;
}

double smoothstep(double edge0, double edge1, double x) {
  // Scale, and clamp x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  // Evaluate polynomial
  return x * x * x * (x * (x * 6 - 15) + 10);
}

/* Fonction qui calcule l'ouverture de la bouche à partir des points de la dlib
  précisément les points 51 à 53 pour la partie supérieur et 57 à 59 pour la partie inférieur */
double lips_distance (const full_object_detection& det)
{
    dlib::vector<double,2> l, r;

    double compt = 0;
    for (unsigned long i = 50; i <= 52; ++i){
        l += det.part(i);
        ++compt;
    }
    /* On prend le point moyen */
    l /= compt;

    compt = 0;
    for (unsigned long i = 56; i <= 58; ++i){
        r += det.part(i);
        ++compt;
    }
    /* On prend le point moyen */
    r /= compt;

    /* On retourne la distance entre les deux points moyen */
    return length(l-r);
}

/* Calcul de "l'ouverture des yeux" */
double eyelid_distance(const full_object_detection& det){
  dlib::vector<double,2> u, d;
  double compt = 0;
  for (unsigned long i = 37; i <= 38; ++i){
      u += det.part(i);
      ++compt;
  }

  for (unsigned long i = 43; i <= 44; ++i){
      u += det.part(i);
      ++compt;
  }
  /* On prend le point moyen */
  u /= compt;

  compt = 0;
  for (unsigned long i = 40; i <= 41; ++i){
      d += det.part(i);
      ++compt;
  }

  for (unsigned long i = 46; i <= 47; ++i){
      d += det.part(i);
      ++compt;
  }
  /* On prend le point moyen */
  d /= compt;

  /* On retourne la distance entre les deux points moyen */
  return length(u-d);
}

double smile_distance(const full_object_detection& det){
  dlib::vector<double,2> u, d;
  u += det.part(33);

  d += det.part(48);
  d += det.part(54);

  d /= 2;


  /* On retourne la distance entre les deux points moyen */
  return length(u-d);
}

double lipsduck_distance(const full_object_detection& det){
  dlib::vector<double,2> l, r;
  l += det.part(48);

  r += det.part(54);

  return length(l-r);
}


double distance_euclidienne(const point p1, const point p2){
   return sqrt(pow(p1.x()-p2.x(),2)+pow(p1.y()-p2.y(),2));
}

double distance_euclidienne(const vec3 p1, const vec3 p2){
   return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2));
}

/* Fonction calculant le point moyen d'un ensemble de landmark */
dpoint gravity_point (const full_object_detection& det, std::vector<int> landmarksId)
{
    dpoint gravityPoint;
    for (unsigned long i = 0; i < landmarksId.size(); ++i){
        gravityPoint += det.part(landmarksId[i]);
    }
    gravityPoint /= landmarksId.size();
    return gravityPoint;
}

/* Fonction retournant un vector de nombre entier contenant toute les valeurs entre 2 valeurs passés en parametre*/
std::vector<int> vector_sequence_maker (int v1, int v2)
{
    std::vector<int> intSequence;
    for (int i = v1; i <= v2; ++i){
        intSequence.push_back(i);
    }
    return intSequence;
}

/* Fonction calculant l'echelle du visage à partir de la largeur(x) des yeux */
float scale (const full_object_detection& det)
{
    dpoint g;
    std::vector<int> allLandmarksId;
    allLandmarksId = vector_sequence_maker(0,67);
    g = gravity_point(det,allLandmarksId);
    if (g.x() > det.part(33).x()){
        return length((det.part(45)-det.part(42))*10/4);
    }else{
        return length((det.part(39)-det.part(36))*10/4);
    }
}

/* Fonction qui calcule la distance entre les sourcils et les yeux à partir des points de la dlib
  précisément les points (18::22),(23::27) pour les sourcils et (37::42),(43::46) pour les yeux */
double eyebrow_distance (const full_object_detection& det)
{
    dlib::vector<double,2> s, y;

    double compt = 0;
    for (unsigned long i = 17; i <= 26; ++i){
        s += det.part(i);
        ++compt;
    }

    /* On prend le point moyen des sourcils */
    s /= compt;

    compt = 0;
    for (unsigned long i = 36; i <= 45; ++i){
        y += det.part(i);
        ++compt;
    }
    /* On prend le point moyen des yeux*/
    y /= compt;

    /* On retourne la distance entre les deux points moyen */
    return length(s-y);
}

/*
  -precond : facteurs doit être le tableau de facteur indexé par l'enum models
*/
void get_facteurs(double* facteurs, full_object_detection shape){

  // cout << "Lips : " << lips_distance(shape)/scale(shape) << " (" << lips_distance(shape) << ")" << endl;
  double temp = lips_distance(shape)/scale(shape) - 0.28f;

  if(temp<0.0f) temp = 0.0f;
  else temp = temp/0.25f;

  if(temp > 1.0f) temp = 1.0f;
  temp = smoothstep(0,1,temp);
  if(abs(temp-facteurs[models::MOUTH_OPEN])>0.1f){
     facteurs[models::MOUTH_OPEN] += temp;
     facteurs[models::MOUTH_OPEN] /= 2.0f;
   } else {
     facteurs[models::MOUTH_OPEN] = temp;
   }


  // cout << "Eyebrow: " << eyebrow_distance(shape)/scale(shape) << " (" << eyebrow_distance(shape) << ")" << endl << endl;
  // cout << "Eyebrow : " << temp << endl << endl;
  if((eyebrow_distance(shape)/scale(shape)-0.28f)>0.0f) {
    temp = (eyebrow_distance(shape)/scale(shape)-0.28f)/0.10f;
    if(temp>1.0f) temp = 1.0f;
    temp = smoothstep(0,1,temp);
    if(abs(temp-facteurs[models::EYEBROW_UP])>0.1f){
     facteurs[models::EYEBROW_UP] += temp;
     facteurs[models::EYEBROW_UP] /= 2.0f;
    } else {
     facteurs[models::EYEBROW_UP] = temp;
    }
    facteurs[models::EYEBROW_DOWN] = 0.0f;
  } else {
    temp = abs(eyebrow_distance(shape)/scale(shape)-0.29f)/0.06f;
    if(temp>1.0f) temp = 1.0f;
    temp = smoothstep(0,1,temp);
    if(abs(temp-facteurs[models::EYEBROW_DOWN])>0.1f){
     facteurs[models::EYEBROW_DOWN] += temp;
     facteurs[models::EYEBROW_DOWN] /= 2.0f;
    } else {
     facteurs[models::EYEBROW_DOWN] = temp;
    }
    facteurs[models::EYEBROW_UP] = 0.0f;
  }

  temp = eyelid_distance(shape)/scale(shape) - 0.018f;

  if(temp<0.0f) temp = 0.0f;
  else temp = temp/0.06f;

  if(temp > 1.0f) temp = 1.0f;
  temp = 1 - temp/(1+0.3*(1-temp));
  if(abs(temp-facteurs[models::EYE_CLOSED])>0.1f){
     facteurs[models::EYE_CLOSED] += temp;
     facteurs[models::EYE_CLOSED] /= 2.0f;
   } else {
     facteurs[models::EYE_CLOSED] = temp;
   }

   cout << "Eyelid: " << eyelid_distance(shape)/scale(shape) << " (" << eyelid_distance(shape) << ")" << " " << temp << endl << endl;


  temp = lipsduck_distance(shape)/scale(shape) - 0.70f;

  if(temp<0.0f) temp = 0.0f;
  else temp = temp/0.28f;

  if(temp > 1.0f) temp = 1.0f;
  temp = 1.0f - smoothstep(0,1,temp);
  if(abs(temp-facteurs[models::DUCKFACE])>0.1f){
    facteurs[models::DUCKFACE] += temp;
    facteurs[models::DUCKFACE] /= 2.0f;
  } else {
    facteurs[models::DUCKFACE] = temp;
  }


  temp = smile_distance(shape)/scale(shape) - 0.09f;

  if(temp<0.0f) temp = 0.0f;
  else temp = temp/0.14f;

  if(temp > 1.0f) temp = 1.0f;
  temp = 1.0f - smoothstep(0,1,temp);
  if(abs(temp-facteurs[models::SMILING])>0.1f){
    facteurs[models::SMILING] += temp;
    facteurs[models::SMILING] /= 2.0f;
  } else {
    facteurs[models::SMILING] = temp;
  }
  // cout << "Lips duck: " << lipsduck_distance(shape)/scale(shape) << " (" << lipsduck_distance(shape) << ")" << endl << endl;
  cout << "Smile: " << smile_distance(shape)/scale(shape) << " (" << smile_distance(shape) << ")" << " " << temp << endl << endl;
  // temp = (unsigned int)();
  // cout << "Smile : " << abs(((length(shape.part(49)-shape.part(58))+length(shape.part(55)-shape.part(58)))/2 - (length(shape.part(49)-shape.part(52))+length(shape.part(55)-shape.part(52)))/2)/scale(shape)) << endl;

}


/* Objets openGL */
Mesh tab_proog[models::SIZE];  //Tableau des modèles
Orbiter camera;
GLuint texture;

/* Objets DLIB et openCV */
cv::VideoCapture* cap;
frontal_face_detector detector;
shape_predictor pose_model;
image_window* win;

/* On enregistre la dernière rotation du visage */
double last_rot = 0;

/* Tableau qui contient les différents facteurs de déformation calculés à partir
    des points du visage détectés pas DLIB */
double* tab_facteur;

int init()
{
/* -------- INITIALISATION DES OBJETS DLIB ET OPENCV  -------- */
  try
  {
      cap = new cv::VideoCapture(0);
      if (!cap->isOpened()){
          cerr << "Impossible d'ouvrir la caméra" << endl;
          return 1;
      }

      // Chargment du fichier de prediction pour l'estimation des points du visage
      detector = get_frontal_face_detector();
      deserialize("data/shape_predictor_68_face_landmarks.dat") >> pose_model;


  }
  catch(serialization_error& e)
  {
      cout << "You need dlib's default face landmarking model file to run this example." << endl;
      cout << "You can get it from the following URL: " << endl;
      cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
      cout << endl << e.what() << endl;
  }
  catch(exception& e)
  {
      cout << e.what() << endl;
  }
/* ---------------------------------------------------------- */

  /* On utilise un tableau qui représente différentes expression du même maillage
      - models::BASE et models::BLEND sont les identiques à l'initialisation mais models::BLEND
        représente le résultat final après les différents pes (le résultat afficher à l'écran)
  */

  tab_proog[models::BASE] = read_mesh("data/proog/ProogBase.obj");
  tab_proog[models::BLEND] = read_mesh("data/proog/ProogBase.obj");
  tab_proog[models::ANGRY] = read_mesh("data/proog/ProogAngry.obj");
  tab_proog[models::DUCKFACE] = read_mesh("data/proog/ProogDuckface.obj");
  tab_proog[models::EYEBROW_UP] = read_mesh("data/proog/ProogEyebrowUp.obj");
  tab_proog[models::EYEBROW_DOWN] = read_mesh("data/proog/ProogEyebrowdown.obj");
  tab_proog[models::EYE_CLOSED] = read_mesh("data/proog/ProogEyeClosed.obj");
  tab_proog[models::MOUTH_OPEN] = read_mesh("data/proog/ProogMouthOpen.obj");
  tab_proog[models::NOSE_UP] = read_mesh("data/proog/ProogNoseUp.obj");
  tab_proog[models::SMILING] = read_mesh("data/proog/ProogSmiling.obj");

  texture= read_texture(0, "data/proog/texture.png");

  tab_facteur = new double[models::SIZE];
  for(int i=0;i<models::SIZE;i++) tab_facteur[i] = 0.0;

  // On crée une caméra qui va englober notre objet
  Point pmin, pmax;
  tab_proog[models::BASE].bounds(pmin, pmax);

  // regle le point de vue de la camera pour observer l'objet
  camera.lookat(pmin, pmax);

  // etat openGL par defaut
  glClearColor(0.2f, 0.2f, 0.2f, 1.f); // couleur par defaut de la fenetre

  // configuration du pipeline.
  /*
    pour obtenir une image correcte lorsque l'on dessine plusieurs triangles, il faut choisir lequel conserver pour chaque pixel...
    on conserve le plus proche de la camera, celui que l'on voit normalement. ce test revient a considerer que les objets sont opaques.
  */
  glClearDepth(1.f);         // profondeur par defaut
  glDepthFunc(GL_LESS);      // ztest, conserver l'intersection la plus proche de la camera
  glEnable(GL_DEPTH_TEST);   // activer le ztest

  return 0;   // ras, pas d'erreur
}

int draw( )
{
/* -------- DETECTON DU VISAGE -------- */
  cv::Mat temp, scaled;
  /* Lecture d'une image sur la webcam */
  if (cap->read(temp)){
    /* On change la taille de l'image pour gagner en perf */
    cv::resize(temp, scaled, cv::Size(320,180));
    // cv::cvtColor(scaled, sca CV_RGB2GRAY);

    /* Conversion pour que l'image soit utilisable par DLIB */
    cv_image<bgr_pixel> cimg_tmp(scaled);
    dlib::array2d<unsigned char> cimg;
    dlib::assign_image(cimg, cimg_tmp);

    /* Détection du visage */
    std::vector<rectangle> faces = detector(cimg);

    /* On sépare les différents visages s'il y en a */
    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));
    if(win){
      win->clear_overlay();
      win->set_image(cimg);
    }

    /* Si un visage a été détecté */
    if(faces.size()){

      {
        /*
          Pose estimation du visage
          pour explication du code voir : https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        */

        std::vector<cv::Point2d> image_points;
        image_points.push_back( cv::Point2d(shapes[0].part(33).x(),shapes[0].part(33).y()));    // Bout du nez
        image_points.push_back( cv::Point2d(shapes[0].part(8).x(), shapes[0].part(8).y()));     // Menton
        image_points.push_back( cv::Point2d(shapes[0].part(36).x(),shapes[0].part(36).y()));    // Oeil gauche, coin gauche
        image_points.push_back( cv::Point2d(shapes[0].part(45).x(),shapes[0].part(45).y()));    // Oeil droit, coin droit
        image_points.push_back( cv::Point2d(shapes[0].part(48).x(),shapes[0].part(48).y()));    // Coin gauche bouche
        image_points.push_back( cv::Point2d(shapes[0].part(54).x(),shapes[0].part(54).y()));    // Coin droit bouche

        // Estimation des points 3D équivalents
        std::vector<cv::Point3d> model_points;
        model_points.push_back(cv::Point3d(0.0f, 133.9f, 810.6f));       // Bout du nez
        model_points.push_back(cv::Point3d(0.0f, -977.6f, 469.4f));      // Menton
        model_points.push_back(cv::Point3d(462.7f, 763.3f, 271.6f));     // Oeil gauche, coin gauche
        model_points.push_back(cv::Point3d(-462.7f, 763.3f, 271.6f));    // Oeil droit, coin droit
        model_points.push_back(cv::Point3d(331.0f, -321.4f, 179.7f));    // Coin gauche bouche
        model_points.push_back(cv::Point3d(-331.0f, -321.4f, 179.7f));   // Coin droit bouche

        // Camera internals
        double focal_length = scaled.cols; // Approximate focal length.
        cv::Point2d center = cv::Point2d(scaled.cols/2,scaled.rows/2);
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

        // Vecteur de rotation et translation
        cv::Mat rotation_vector;
        cv::Mat translation_vector;

        // Résolution Perspective-n-Point de openCV
        solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

        Point pmin, pmax;
        tab_proog[models::BLEND].bounds(pmin, pmax);

        camera.lookat(pmin, pmax);
        /* on ignore les micros rotations */
        // if(rotation_vector.at<double>(1,0)-last_rot<0.1f){
        //   camera.rotation(rotation_vector.at<double>(1,0)*100, 0);
        //   last_rot = rotation_vector.at<double>(1,0);
        // } else {
        //   last_rot = (last_rot + rotation_vector.at<double>(1,0))/2;
        //   camera.rotation(last_rot*100, 0);
        // }
      }


      /* On calcule le facteur de déformation seulement pour l'ouverture de la bouche dans ce cas */
      if(faces.size()){
        get_facteurs(tab_facteur, shapes[0]);
      }
      if(win) win->add_overlay(render_face_detections(shapes));
    }
  }
/* ------------------------------------- */


/* -------- BLENDSHAPE -------- */
  std::vector<vec3> base = tab_proog[models::BASE].positions();
  std::vector<vec3> def = tab_proog[models::MOUTH_OPEN].positions();
  std::vector<vec3> obj = base;

  // Blendshape en fonction du facteur de déformation
  for(unsigned int i = 0;i<def.size();i++){
    vec3 p;
    p.x = base[i].x +
          (tab_facteur[models::MOUTH_OPEN])*(tab_proog[models::MOUTH_OPEN].positions()[i].x - base[i].x) +
          (tab_facteur[models::EYEBROW_UP])*(tab_proog[models::EYEBROW_UP].positions()[i].x - base[i].x) +
          (tab_facteur[models::EYEBROW_DOWN])*(tab_proog[models::EYEBROW_DOWN].positions()[i].x - base[i].x) +
          (tab_facteur[models::EYE_CLOSED])*(tab_proog[models::EYE_CLOSED].positions()[i].x - base[i].x) +
          (tab_facteur[models::DUCKFACE])*(tab_proog[models::DUCKFACE].positions()[i].x - base[i].x) +
          (tab_facteur[models::SMILING])*(tab_proog[models::SMILING].positions()[i].x - base[i].x);
    p.y = base[i].y +
          (tab_facteur[models::MOUTH_OPEN])*(tab_proog[models::MOUTH_OPEN].positions()[i].y - base[i].y) +
          (tab_facteur[models::EYEBROW_UP])*(tab_proog[models::EYEBROW_UP].positions()[i].y - base[i].y) +
          (tab_facteur[models::EYEBROW_DOWN])*(tab_proog[models::EYEBROW_DOWN].positions()[i].y - base[i].y) +
          (tab_facteur[models::EYE_CLOSED])*(tab_proog[models::EYE_CLOSED].positions()[i].y - base[i].y) +
          (tab_facteur[models::DUCKFACE])*(tab_proog[models::DUCKFACE].positions()[i].y - base[i].y) +
          (tab_facteur[models::SMILING])*(tab_proog[models::SMILING].positions()[i].y - base[i].y);
    p.z = base[i].z +
          (tab_facteur[models::MOUTH_OPEN])*(tab_proog[models::MOUTH_OPEN].positions()[i].z - base[i].z) +
          (tab_facteur[models::EYEBROW_UP])*(tab_proog[models::EYEBROW_UP].positions()[i].z - base[i].z) +
          (tab_facteur[models::EYEBROW_DOWN])*(tab_proog[models::EYEBROW_DOWN].positions()[i].z - base[i].z) +
          (tab_facteur[models::EYE_CLOSED])*(tab_proog[models::EYE_CLOSED].positions()[i].z - base[i].z) +
          (tab_facteur[models::DUCKFACE])*(tab_proog[models::DUCKFACE].positions()[i].z - base[i].z) +
          (tab_facteur[models::SMILING])*(tab_proog[models::SMILING].positions()[i].z - base[i].z);
    tab_proog[models::BLEND].vertex(i, p);
  }
/* ----------------------------- */

/* -------- DESSIN OPENGL ET GESTION DE LA SOURIS  -------- */

  // on commence par effacer la fenetre avant de dessiner quelquechose
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // recupere les mouvements de la souris, utilise directement SDL2
  int mx, my;
  unsigned int mb= SDL_GetRelativeMouseState(&mx, &my);
  if(mb & SDL_BUTTON(1)){ // le bouton gauche est enfonce
    // tourne autour de l'objet
    camera.rotation(mx, my);
  } else if(mb & SDL_BUTTON(3)){ // le bouton droit est enfonce
    // approche / eloigne l'obje
    camera.move(mx);
  } else if(mb & SDL_BUTTON(2)){// le bouton du milieu est enfonce
    // deplace le point de rotation
    camera.translation((float) mx / (float) window_width(), (float) my / (float) window_height());
  }

  /* Gestion du clavier */
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  if(state[SDL_SCANCODE_C]) {
    if(!win) win = new image_window();
  }

  /* On dessine le maillage déformé du point de vu de notre caméra */
  draw(tab_proog[models::BLEND], camera);

/* --------------------------------------------------------- */

  return 1;
}

int quit( )
{
  // Destruction des objets
  tab_proog[models::BASE].release();
  tab_proog[models::BLEND].release();
  tab_proog[models::ANGRY].release();
  tab_proog[models::DUCKFACE].release();
  tab_proog[models::EYEBROW_UP].release();
  tab_proog[models::EYEBROW_DOWN].release();
  tab_proog[models::EYE_CLOSED].release();
  tab_proog[models::MOUTH_OPEN].release();
  tab_proog[models::NOSE_UP].release();
  tab_proog[models::SMILING].release();

  delete win;
  delete cap;
  delete tab_facteur;

  return 0;
}


int main( int argc, char **argv )
{
  // Création de la fenêtre
  Window window= create_window(1024, 640);
  if(window == NULL)
      return 1;

  // Création du context openGL
  Context context= create_context(window);
  if(context == NULL)
      return 1;

  // Appel de la fonction init
  if(init() < 0)
  {
    cerr << "[error] échec de l'initialisation" << endl;
    return 1;
  }

  // Lancement de l'application sur la fenêtre qu'on vient de créer
  run(window, draw);


  // Nettoyage
  quit();
  release_context(context);
  release_window(window);
  return 0;
}
