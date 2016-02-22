/*
 * FreeGLUT Shapes Demo
 *
 * Written by Nigel Stewart November 2003
 *
 * This program is test harness for the sphere, cone 
 * and torus shapes in FreeGLUT.
 *
 * Spinning wireframe and smooth shaded shapes are
 * displayed until the ESC or q key is pressed.  The
 * number of geometry stacks and slices can be adjusted
 * using the + and - keys.
 */

#include <GL/glut.h>

#include <stdlib.h>
#include <math.h>


#define L 2000.0
#define H 2000.0
#define W 2000.0

#define DT 0.1

#define NX 20
#define NY 20

#define DX (L/NX)
#define DY (H/NY)

class BODY {
    
    float x, y, z;   // Reference center location
    float BL, BH, BW;  // Size
    float size_vec[3];
    
    public: 
        int test(float, float, float, float, float, float);
        void draw();
        void init();
    
} body[2];


// Introduce the bullet class
class BULLET {
    bool draw_flag;      // This is the flag to draw the bullet, which means it's moving
    float x,y,z;
    float x_ref, y_ref, z_ref;
    float vx,vy,vz;
    double fire_time;
    
    public:
        void init(float, float, float);
        void draw();
        void move();
        void fire(float, float, float);
}; 


// Introduce the cannon class
class CANNON {
    float x, y, z; // Location on x-y plane
    float fire_angle; // Angle with the x-y plane
    float direction_angle; // Direction the cannon is facing
    BULLET bullet;         // Each cannon has its own ammo
    public:
        void draw();
        void init(float, float, float);
        void turn(float);     
        void tilt(float);  
        void fire();
} cannon[2];


static int slices = 16;
static int stacks = 16;

float ROTATE[] = {-60.0, 0.0, 0.0};
float OBJECT[] = {0.0, 0.0, -5.0};
float WIND[] = {10.0, 10.0, 0.0};
float SCALE  = 0.005;
/* GLUT callback Handlers */

static void resize(int width, int height)
{
    const float ar = (float) width / (float) height;
    
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity() ;
}

static void 
display(void)
{
    const double t = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    const double a = t*90.0;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3d(1,0,0);

   glPushMatrix();

        //glTranslated(0.0,0.0,GROUND);
        glTranslatef(OBJECT[0],OBJECT[1],OBJECT[2]);
        //glLoadIdentity() ;
        // Rotate our grid
        glRotatef(ROTATE[0], 1.0, 0.0, 0.0);
        glRotatef(ROTATE[1], 0.0, 1.0, 0.0);
        glRotatef(ROTATE[2], 0.0, 0.0, 1.0);    
        glScalef(SCALE, SCALE, SCALE);
  
        // Draw Ground
        //glColor3d(0,0,0);
        for (int i = 0; i <= NX; i++) {
            glBegin(GL_LINES);
                // Draw each line in the Y direction
                glVertex3f( (i-0.5*NX)*DX, -0.5*H, 0.0);
                glVertex3f( (i-0.5*NX)*DX, 0.5*H, 0.0);
            glEnd();    
        }
        for (int i = 0; i <= NY; i++) {
            glBegin(GL_LINES);
                // Draw each line in the Y direction
                glVertex3f( -0.5*L, (i-0.5*NY)*DY, 0.0);
                glVertex3f( 0.5*L, (i-0.5*NY)*DY, 0.0);
            glEnd();    
        }
    glPopMatrix();

    // Draw a cannon
    glColor3d(1,0,0);
    cannon[0].draw();

    glColor3d(0,1,0);
    cannon[1].draw();

    
    // Draw the body
    glColor3d(1,0,0);
    body[0].draw();

    glutSwapBuffers();
}


static void 
key(unsigned char key, int x, int y)
{
    switch (key) 
    {
         case 27 : 
            exit(0);
            break;

        case '+':
            SCALE += 0.0001;
            break;

        case '-':
            SCALE -= 0.0001;
            break;

       case '4':
            ROTATE[2] += 1.0;
            break;
        case '6':
            ROTATE[2] -= 1.0;
            break;

        case '8':
            ROTATE[0] += 1.0;
            break;
        case '2':
            ROTATE[0] -= 1.0;
            break;

        // Cannon 1 control
        case 'e':
            cannon[0].fire();
            break;
        
        case 'a':
            cannon[0].turn(2.0);
            break;
        case 'd':
            cannon[0].turn(-2.0);
            break;

        case 'w':
            cannon[0].tilt(2.0);
            break;
        case 's':
            cannon[0].tilt(-2.0);
            break;
        
        // Cannon 2 controls

        case 'o':
            cannon[1].fire();
            break;

        case 'j':
            cannon[1].turn(2.0);
            break;
        case 'l':
            cannon[1].turn(-2.0);
            break;

        case 'i':
            cannon[1].tilt(2.0);
            break;
        case 'k':
            cannon[1].tilt(-2.0);
            break;
      
     
    }

    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
   // Wheel reports as button 3(scroll up) and button 4(scroll down)
   if ((button != 1) && (button != 2)) // It's a wheel event
   {
       // SCALE += 0.001;
        
       if (button == 3) {
            // Zoom in - increase the scale
            SCALE += 0.001;
       } else {
            // Zoom out - decrease the scale
            SCALE -= 0.001;
        }
   } else { 
       // normal button event
   }
   // The display needs reposting
   glutPostRedisplay();
}




static void 
idle(void)
{
    glutPostRedisplay();
}

const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

/* Program entry point */

int 
main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow("FreeGLUT Shapes");

    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutMouseFunc(mouse);
    glutIdleFunc(idle);

    glClearColor(1,1,1,1);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);

    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

    // Initialise our objects
    cannon[0].init(-0.25*L, -0.25*H, 0.0);
    cannon[1].init(0.25*L, 0.25*H, 0.0);
    
    // Initialise our bodies
    body[0].init();

    glutMainLoop();

    return EXIT_SUCCESS;
}

void BODY::init() {
    
    // Set the location and size
    x = 0.0;
    y = 0.0;
    z = 0.0;
    size_vec[0] = 0.5;
    size_vec[1] = 0.2;
    size_vec[2] = 0.2; 
    BL = size_vec[0]*L;
    BH = size_vec[1]*H;
    BW = size_vec[2]*W;
}

void BODY::draw() {
 
    // Draw the body
    glPushMatrix();

        // Move to the center of the region
        glTranslatef(OBJECT[0],OBJECT[1],OBJECT[2]);  
        glRotatef(ROTATE[0], 1.0, 0.0, 0.0);
        glRotatef(ROTATE[1], 0.0, 1.0, 0.0);
        glRotatef(ROTATE[2], 0.0, 0.0, 1.0);    
        glScalef(SCALE, SCALE, SCALE);     
        
        // Now, draw our shape
        // It's special
        glPushMatrix();
            glTranslatef(x, y, z);
            glScalef(size_vec[0], size_vec[1], size_vec[2]);     
            glutSolidCube(L);
        glPopMatrix();
     glPopMatrix();  
}
 
int BODY::test(float x_ref, float y_ref, float z_ref, float vx, float vy, float vz) {
    // Check to see if this falls into the body region
    // Return 1 to reflect x
    // Return 2 to reflect y
    // Return 3 to reflect z
    if  ( (x_ref > (x - 0.5*BL)) && (x_ref < (x + 0.5*BL)) &&
          (y_ref > (y - 0.5*BH)) && (y_ref < (y + 0.5*BH)) &&
          (z_ref > (z - 0.5*BW)) && (z_ref < (z + 0.5*BW))  ) {
    
        // We are indeed inside the object. Let's find out which edge is correct.
        x_ref = x_ref - vx*DT;
        if ((x_ref <= (x - 0.5*BL)) || (x_ref >= (x + 0.5*BL))) {
            // Good enough. Reverse the x velocity
            return 1;
        }
        y_ref = y_ref - vy*DT;
        if ((y_ref <= (y - 0.5*BH)) || (y_ref >= (y + 0.5*BH))) {
            return 2;
        }
        z_ref = z_ref - vz*DT;
        if ((z_ref <= (z - 0.5*BW)) || (z_ref >= (z + 0.5*BW))) {
            return 3;
        }
    } else {
        return 0;
    }
}

void BULLET::init(float xr, float yr, float zr) {
    x = 0.0; 
    y = 0.0; 
    z = 90.0; 
    vx = 0.0;
    vy = 0.0;
    vz = 0.0;
    x_ref = xr;
    y_ref = yr;
    z_ref = zr;
    draw_flag = false;
}

void BULLET::fire(float vel, float alpha, float beta) {
 
     // Fire the bullet
    if (draw_flag) init(x_ref, y_ref, z_ref);

    draw_flag = true;
    alpha = (alpha-90.0)*M_PI/180.0;
    beta = (beta)*M_PI/180.0;
    vx = vel*cos(beta)*cos(alpha);
    vy = vel*cos(beta)*sin(alpha);
    vz = vel*sin(beta);    
    fire_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
   
}

void BULLET::move() {

    const double current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    float ax, ay, az;
    float relative_speed;
    float dragx, dragy, dragz;
    float x_global, y_global, z_global;
    int body_check;
    
    // Need to update vx, vy and vz
    relative_speed = vx - WIND[0];
    dragx = 0.001*relative_speed*fabs(relative_speed);
    relative_speed = vy - WIND[1];
    dragy = 0.001*relative_speed*fabs(relative_speed);
    relative_speed = vz - WIND[2];
    dragz = 0.001*relative_speed*fabs(relative_speed);

    ax = -dragx;
    ay = -dragy;
    az = -dragz -9.81;
    vx = vx + ax*DT;
    vy = vy + ay*DT;
    vz = vz + az*DT;
    
    x = x + vx*DT;
    y = y + vy*DT;
    z = z + vz*DT;     
    
    // Calculate the global positions too
    x_global = x + x_ref; 
    y_global = y + y_ref;
    z_global = z + z_ref; 
    
    // Check for body intersections
    body_check = body[0].test(x_global, y_global, z_global, vx, vy, vz);
    
        
    // if the bullet hits the ground, bounce it
    if ((z_global < 0.0) || (body_check == 3)) {
        vz = -vz;
        z = z + 1.5*vz*DT;
    }

    // WALLS     
    if ((x_global < -0.5*L) || (x_global > 0.5*L) || (body_check == 1)) {
        vx = -vx;
        x = x + vx*DT;
    }
    if ((y_global < -0.5*H) || (y_global > 0.5*H) || (body_check == 2)) {
        vy = -vy;
        y = y + vy*DT;
    }

    if ((current_time - fire_time) > 10.0) {
        // Cancel the whole thing
        init(x_ref, y_ref, z_ref); 
    }
}

void BULLET::draw() {
    
    if (draw_flag) {
        // We have to draw it and move it
        // Move it first
        move();
        glPushMatrix();
            glTranslatef(x, y, z);
            glutSolidSphere(10.0,10,10);   
        glPopMatrix();    
    }
}

void CANNON::fire() {
    
    bullet.fire(200.0, direction_angle, fire_angle);    
    
}

void CANNON::init(float x_init, float y_init, float z_init) {
    // Set the diplacement vectors from the center of the field
    x = x_init; 
    y = y_init;
    z = z_init;
    fire_angle = 0.0;
    direction_angle = 0.0;
    // Need to init the bullet too
    bullet.init(x,y,z);
}

void CANNON::turn(float change_in_angle) {
    direction_angle += change_in_angle;
}

void CANNON::tilt(float change_in_angle) {
    fire_angle += change_in_angle;
}

void CANNON::draw() {
    
    // Draw the cannon in the correct place
    glPushMatrix();

        glTranslatef(OBJECT[0],OBJECT[1],OBJECT[2]);  
        glRotatef(ROTATE[0], 1.0, 0.0, 0.0);
        glRotatef(ROTATE[1], 0.0, 1.0, 0.0);
        glRotatef(ROTATE[2], 0.0, 0.0, 1.0);    
        glScalef(SCALE, SCALE, SCALE);
        
        // Move to the center
        glTranslatef(x,y,z); 

        // Now is a good time to draw the bullets before we perform our rotation
        // Now to draw the bullets from the cannon
        bullet.draw(); 

        // Face the direction of fire
        glRotatef(direction_angle, 0.0, 0.0, 1.0);
        
        // Draw each item
        glPushMatrix(); 
            glTranslatef(-20.0, 0.0, 0.0);
            glutSolidCone(10.0,100.0,10,10);   
        glPopMatrix();

        glPushMatrix(); 
            glTranslatef(+20.0, 0.0, 0.0);
            glutSolidCone(10.0,100.0,10,10);   
        glPopMatrix();

        // Draw the cannon body
        glPushMatrix(); 
            glTranslatef(0.0, 0.0, 90.0);
            glRotatef(90.0 - fire_angle, 1.0, 0.0, 0.0);
            glTranslatef(0.0, 0.0, -40.0);
            glutSolidCone(20.0,100.0,10,10);   
            //glRotatef(90.0 - fire_angle, 1.0, 0.0, 0.0);
        glPopMatrix();

    glPopMatrix();
    

}

