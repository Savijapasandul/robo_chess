import com.jogamp.newt.opengl.GLWindow;

// controls
CameraControl cam;


Arm myArm;


//UI
PFont UIFont;
int ui_font_size = 25;
ArrayList<TextBox> tb_list = new ArrayList<TextBox>();


void setup() {
  size(1200,900,P3D);
  // font setup
  UIFont = createFont("Consolas", ui_font_size);
  textFont(UIFont);
  
  cam = new CameraControl(this);
  myArm = new Arm();
  
  // UI stuff
  // x y width height
  for (int i = 1; i <= 6; i++){
    TextBox newServ = new TextBox(width - 70, height - (50 * i), 100, 20, ui_font_size);
    tb_list.add(newServ);
  }
  /*
  Slider test_s = new Slider();
  println(test_s.getValuePercent());
  */
}

void draw() {
  background(0);
  perspective();
  
  stroke(255, 100, 100);
  line(-500, 0, 0, 500, 0, 0);
  line(0, -500, 0, 0, 500, 0);
  line(0, 0, -500, 0, 0, 500);
  noStroke();
  stroke(255, 200, 200);
  myArm.drawArm();
  //camera(60, height/2.5, (height/8), 0, height/2, 0, 0, 1, 0);
  
  UIDrawControl();
}

void mousePressed(){
  for (TextBox servo_ui : tb_list){
    servo_ui.checkFocus();
    if (servo_ui.focused && cam.enabled){
      cam.enabled = false;
    }
    else if (!servo_ui.focused && !cam.enabled){
      cam.enabled = true;
    }
  }
}

void UIDrawControl(){
  // hud
  pushMatrix();
  ortho();
  resetMatrix();
  rectMode(LEFT);
  textAlign(LEFT);
  translate(-width/2.0, -height/2.0);
  hint(DISABLE_DEPTH_TEST);
  fill(255);
  text("[UP],[DOWN] : Tilt up/down", 20, height - 20);
  text("[LEFT],[RIGHT] : Pan left/right", 20, height - 50);
  text("[w],[s] : Move forward/backward", 20, height - 80);
  text("[a],[d] : Move left/right", 20, height - 110);
  text("[e],[c] : Move up/down", 20, height - 140);
  text("[h] to update arm", 20, height - 170);
  text("Servo Limits: 500-2500", 10, 30);
  hint(ENABLE_DEPTH_TEST);
  popMatrix();
  
  // UI
  for (TextBox servo_ui : tb_list){
    servo_ui.drawTextBox();
    //servo_ui.drawTextBox();
  }
}

void keyPressed(){
  for (TextBox servo_ui : tb_list){
    if (servo_ui.focused){
      servo_ui.editText();
      if (key == ENTER){
        cam.enabled = true;
      }
    }
  }
  if (key == 'h' && cam.enabled){
    for (int i = 0; i <= 5; i++){
      int ui_value = tb_list.get(i).getValue();
      updateMotor(i, ui_value);
    }
  }
  
}

// make rotation inputs match what arm should be
void updateMotor(int index, int value){
  if (myArm.rotation_range[0] <= value && value <= myArm.rotation_range[1]){
    myArm.action(index, value);
  }
}
