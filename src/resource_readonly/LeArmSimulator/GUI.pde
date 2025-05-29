class Slider{
  int value = 1500;
  int[] value_range = {500, 2500};
  // x, y on screen
  float[] screen_pos = {100, 100};
  float w = 50;
  float h = 50;
  SliderDragBox sliding_box = new SliderDragBox(screen_pos[0], screen_pos[1], w / 10, h);
  SliderBackground sliding_bg = new SliderBackground(screen_pos[0], screen_pos[1], w, h);

  public float getValuePercent(){
    // median is needed because arm works where anything 500-1500 is one direction and 1500-2500 is the other
    float median = (this.value_range[0] + this.value_range[1]) / 2;
    float value_percent = (value - median) / (value_range[1] - median);
    return value_percent;
  }
  
  public void drawSlider(){
    this.sliding_box.drawDragBox();
    this.sliding_bg.drawBackground();
  }

}


class SliderDragBox{
  float x = 0;
  float y = 0;
  float w = 5;
  float h = 10;
  
  public SliderDragBox(float x, float y, float w, float h){
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
  }
  
  public void drawDragBox(){
    pushMatrix();
    ortho();
    resetMatrix();
    rectMode(CENTER);
    rect(this.x, this.y, this.w, this.h);
    popMatrix();
  }
}


class SliderBackground{
  float x = 0;
  float y = 0;
  float w = 0;
  float h = 0;

  public SliderBackground(float x, float y, float w, float h){
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
  }
  
  public void drawBackground(){
    pushMatrix();
    ortho();
    resetMatrix();
    rectMode(CENTER);
    // center line (slider along here)
    rect(this.x + (this.w / 2), this.y, this.w, this.h / 5);
    // left bounding box
    rect(this.x, this.y, this.w / 10, this.h);
    // right bounding box
    rect(this.x + this.w, this.y, this.w / 10, this.h);
    popMatrix();
    
  }
}


class TextBox{
  float x = 0;
  float y = 0;
  float w = 0;
  float h = 0;
  String text = "1500";
  int font_size = 12;
  boolean focused = false;
  
  public TextBox(float x, float y, float w, float h, int font_size){
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.font_size = font_size;
  }
  
  public int getValue(){
    int output = Integer.parseInt(this.text);
    return output;
  }
  
  public void checkFocus(){
      // if mouse not within x boundary
    if ((this.x - this.w / 2) <= mouseX && mouseX <= (this.x + this.w / 2)){
      // if mouse not within y boundary
      if ((this.y - this.h / 2) <= mouseY && mouseY <= (this.y + this.h / 2)){
        this.focused = true;
        return;
      }
    }
    this.focused = false;
  }
  
  public void editText(){
    // if mouse isnt pressed box isnt edited
    if (!focused){
      return;
    }
    // this code executes if mouse is within boundary
    // if backspace
    if (key == BACKSPACE){
      if (this.text != null && this.text.length() > 0) {
        this.text = this.text.substring(0, this.text.length() - 1);
      }
    }
    // if enter lose focus
    else if (key == ENTER){
      this.focused = false;
    }
    
    // if numbers entered, add to string
    else if ((48 <= key) && (key <= 57)){
      this.text = this.text + key;
    }
  }
  
  public void drawTextBox(){
    this.w = this.text.length() * 7 * (this.font_size / 8);
    if (this.text.length() == 1){
      this.w += 2;
    }
    if (this.text.length() <= 0 ){
      return;
    }
    this.h = (this.font_size / 4) + this.font_size;
    pushMatrix();
    resetMatrix();
    rectMode(CENTER);
    textAlign(CENTER, CENTER);
    ortho();
    translate(-width/2.0, -height/2.0);
    // text(str, x, y, w, h)
    fill(155);
    rect(this.x, this.y, this.w, this.h);
    fill(255);
    text(this.text, this.x, this.y, this.w, this.h);
    popMatrix();
  }
}
