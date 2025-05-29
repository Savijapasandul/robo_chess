// sizes of each arm piece
float[] size_pl = {0, 0, 0};
float[] size_seg6_5 = {50, 50, 50};
float[] size_seg5_4 = {10, 50, 25};
float[] size_seg4_3 = {10, 45, 15};
float[] size_seg3_2 = {10, 25, 25};
float[] size_seg2_1 = {15, 15, 5};
float[] size_seg1 = {5, 20, 7};
float[] size_seg1base = {15, 15, 20};

//initial position values for each arm piece
mathFinder mf = new mathFinder();
float[] pos_pl = {0, 0, 0};
float[] pos_seg6_5 = {0, 0, 0};
float[] pos_seg5_4 = {0, mf.getPosY(pos_seg6_5, size_seg6_5, size_seg5_4), 0};
float[] pos_seg4_3 = {0, mf.getPosY(pos_seg5_4, size_seg5_4, size_seg4_3), 0};
float[] pos_seg3_2 = {0, mf.getPosY(pos_seg4_3, size_seg4_3, size_seg3_2), 0};
float[] pos_seg2_1 = {0, mf.getPosY(pos_seg3_2, size_seg3_2, size_seg2_1), 0};
float[] pos_seg1base = {0, mf.getPosY(pos_seg2_1, size_seg2_1, size_seg1base), 0};
float[] pos_seg1 = {0, mf.getPosY(pos_seg1base, size_seg1base, size_seg1), 0};

float[][] all_size = {size_pl, size_seg6_5, size_seg5_4, size_seg4_3, size_seg3_2, size_seg2_1, size_seg1, size_seg1base};
float[][] all_pos = {pos_pl, pos_seg6_5, pos_seg5_4, pos_seg4_3, pos_seg3_2, pos_seg2_1, pos_seg1, pos_seg1base}; 


class mathFinder{
  public float getPosY(float[] prev_seg_pos, float[] prev_seg_size, float[] seg_size){
    return prev_seg_pos[1] - (prev_seg_size[1] / 2) - (seg_size[1] / 2);
  }
  
  public float[] getArmPiecePos(ArmPiece parent, ArmPiece child){
    float x_child = sin(child.total_rotation[2]) * (child.size[1] / 2) + sin(parent.total_rotation[2]) * (parent.size[1] / 2);
    float y_child = - cos(child.total_rotation[2]) * (child.size[1] / 2) - cos(parent.total_rotation[2]) * (parent.size[1] / 2);
    float z_child = sin(child.total_rotation[1]) * (x_child);
    x_child *= cos(child.total_rotation[1]);
    float x = x_child + parent.x;
    float y = y_child + parent.y;
    float z = -z_child + parent.z;
    float[] final_pos = {x, y, z};
    
    return final_pos;
  }
  
  public float[] getArmPinchersPos(ArmPiece parent, ArmPinchers child){
    float x_child = sin(child.total_rotation[2]) * (child.size[1] / 2) + sin(parent.total_rotation[2]) * (parent.size[1] / 2);
    float y_child = - cos(child.total_rotation[2]) * (child.size[1] / 2) - cos(parent.total_rotation[2]) * (parent.size[1] / 2);
    float z_child = sin(child.total_rotation[1]) * (x_child);
    x_child *= cos(child.total_rotation[1]);
    float x_difference = sin(parent.total_rotation[1]) * ((parent.size[2] / 2));
    float z_difference = cos(parent.total_rotation[1]) * ((parent.size[2] / 2));
    float x = x_child + parent.x;
    float y = y_child + parent.y;
    float z = -z_child + parent.z;
    float[] final_pos = {x + x_difference, y, z + z_difference, x - x_difference, y, z - z_difference};
    
    return final_pos;
  }
}


class ArmPiece{
  float x = 0;
  float y = 0;
  float z = 0;
  float[] size = {10, 10, 10};
  float[] rotation = {0, 0, 0};
  float[] total_rotation = {0 ,0, 0};
  float[] rotation_range = {500, 2500};
  ArmPiece parent;
  int rotation_dir = 2;
  boolean moved = false;

  public ArmPiece(float x, float y, float z, float[] size, float[] rotation, ArmPiece parent, int rotation_dir){
    this.x = x;
    this.y = y;
    this.z = z;
    this.size = size;
    this.rotation[0] = rotation[0];
    this.rotation[1] = rotation[1];
    this.rotation[2] = rotation[2];
    this.parent = parent;
    this.rotation_dir = rotation_dir;
  }
  
  public void drawPiece(){
    if (this.parent.moved | this.moved){
      float[] r = {this.parent.total_rotation[0] + this.rotation[0], this.parent.total_rotation[1] + this.rotation[1], this.parent.total_rotation[2] + this.rotation[2]};
      this.total_rotation = r;
      this.moved = true;
      this.parent.moved = false;
    }
    pushMatrix();
    float[] piece_pos = mf.getArmPiecePos(this.parent, this);
    this.x = piece_pos[0];
    this.y = piece_pos[1];
    this.z = piece_pos[2]; //<>//
    translate(this.x, this.y, this.z);
    rotateX(this.total_rotation[0]);
    rotateY(this.total_rotation[1]);
    rotateZ(this.total_rotation[2]);
    fill(100);
    box(this.size[0], this.size[1], this.size[2]);
    popMatrix();
  }
  
  public void move(float to_angle){
    // don't bother moving if angle is the same
    if (this.rotation[this.rotation_dir] == to_angle){
      return;
    }
    this.rotation[this.rotation_dir] = to_angle;
    this.moved = true;
  }
}


class ArmPieceBuilder{
  private float x = 0;
  private float y = 0;
  private float z = 0;
  private float[] size = {10, 10, 10};
  private float[] rotation = {0, 0, 0};
  private float[] rotation_range = {500, 2500};
  private ArmPiece parent;
  private int rotation_dir = 2;
  
  public ArmPieceBuilder(float x, float y, float z, float[] size, float[] rotation){
    this.x = x;
    this.y = y;
    this.z = z;
    this.size = size;
    this.rotation = rotation;
  }
  
  ArmPieceBuilder setParent(ArmPiece parent){
    this.parent = parent;
    return this;
  }
  
  ArmPieceBuilder setRotationDir(int rotation_dir){
    this.rotation_dir = rotation_dir;
    return this;
  }
  
  ArmPiece build(){
    return new ArmPiece(x, y, z, size, rotation, parent, rotation_dir);
  }
}


class ArmPinchers extends ArmPiece{
  ArmPiece piece1;
  ArmPiece piece2;

  public ArmPinchers(float x, float y, float z, float[] size, float[] rotation, ArmPiece parent, int rotation_dir){
    super(x, y, z, size, rotation, parent, rotation_dir);
    piece1 = new ArmPieceBuilder(x, y, z + (parent.size[2] / 2) + (size[2] / 2), size, rotation).setParent(parent).setRotationDir(rotation_dir).build();
    piece2 = new ArmPieceBuilder(x, y, z - (parent.size[2] / 2) - (size[2] / 2), size, rotation).setParent(parent).setRotationDir(rotation_dir).build();
  }
  
  public void drawPiece(){
    if (this.parent.moved | this.moved){
      float[] r = {this.parent.total_rotation[0] + this.rotation[0], this.parent.total_rotation[1] + this.rotation[1], this.parent.total_rotation[2] + this.rotation[2]};
      this.total_rotation = r;
      this.moved = true;
      this.parent.moved = false;
      this.piece1.total_rotation = r;
      this.piece2.total_rotation = r;
    }
    float[] piece_pos = mf.getArmPinchersPos(this.parent, this);
    pushMatrix();
    this.x = piece_pos[0];
    this.y = piece_pos[1];
    this.z = piece_pos[2];
    translate(this.x, this.y, this.z);
    rotateX(this.total_rotation[0]);
    rotateY(this.total_rotation[1]);
    rotateZ(this.total_rotation[2]);
    fill(100);
    box(this.size[0], this.size[1], this.size[2]);
    popMatrix();
    
    pushMatrix();
    this.x = piece_pos[3];
    this.y = piece_pos[4];
    this.z = piece_pos[5];
    translate(this.x, this.y, this.z);
    rotateX(-this.total_rotation[0]);
    rotateY(this.total_rotation[1]);
    rotateZ(this.total_rotation[2]);
    fill(100);
    box(this.size[0], this.size[1], this.size[2]);
    popMatrix();
  }
  
  public void move(float to_angle){
    this.rotation[this.rotation_dir] = to_angle;
    this.moved = true;
  }
}


class ArmPinchersBuilder{
  private float x = 0;
  private float y = 0;
  private float z = 0;
  private float[] size = {10, 10, 10};
  private float[] rotation = {0, 0, 0};
  private float[] rotation_range = {500, 2500};
  private ArmPiece parent;
  private int rotation_dir = 2;
  
  public ArmPinchersBuilder(float x, float y, float z, float[] size, float[] rotation){
    this.x = x;
    this.y = y;
    this.z = z;
    this.size = size;
    this.rotation = rotation;
  }
  
  ArmPinchersBuilder setParent(ArmPiece parent){
    this.parent = parent;
    return this;
  }
  
  ArmPinchersBuilder setRotationDir(int rotation_dir){
    this.rotation_dir = rotation_dir;
    return this;
  }
  
  ArmPinchers build(){
    return new ArmPinchers(x, y, z, size, rotation, parent, rotation_dir);
  }
}


class Arm{
  float[] start_rotation = {0, 0, 0};
  float[] pl_rotation = {0, 0, 0};
  float[] rotation_range = {500, 2500};
  // placeholder arm for pieces that don't need
  ArmPiece placeholder = new ArmPieceBuilder(pos_pl[0], pos_pl[1], pos_pl[2], size_pl, pl_rotation).build();
  // create pieces of arm
  ArmPiece base = new ArmPieceBuilder(pos_seg6_5[0], pos_seg6_5[1], pos_seg6_5[2], size_seg6_5, start_rotation).setParent(placeholder).setRotationDir(1).build();
  ArmPiece s5_4 = new ArmPieceBuilder(pos_seg5_4[0], pos_seg5_4[1], pos_seg5_4[2], size_seg5_4, start_rotation).setParent(base).build();
  ArmPiece s4_3 = new ArmPieceBuilder(pos_seg4_3[0], pos_seg4_3[1], pos_seg4_3[2], size_seg4_3, start_rotation).setParent(s5_4).build();
  ArmPiece s3_2 = new ArmPieceBuilder(pos_seg3_2[0], pos_seg3_2[1], pos_seg3_2[2], size_seg3_2, start_rotation).setParent(s4_3).build();
  ArmPiece s2_1 = new ArmPieceBuilder(pos_seg2_1[0], pos_seg2_1[1], pos_seg2_1[2], size_seg2_1, start_rotation).setParent(s3_2).setRotationDir(1).build();
  ArmPiece seg1base = new ArmPieceBuilder(pos_seg1base[0], pos_seg1base[1], pos_seg1base[2], size_seg1base, start_rotation).setParent(s2_1).build();
  ArmPinchers seg1 = new ArmPinchersBuilder(pos_seg1[0], pos_seg1[1], pos_seg1[2], size_seg1, start_rotation).setParent(seg1base).setRotationDir(0).build();
  
  ArmPiece[] pieces = {base, s5_4, s4_3, s3_2, s2_1, seg1, seg1base};
  
  public void drawArm(){
    for (ArmPiece piece : this.pieces){
      piece.drawPiece();
    }
  }
  
  public boolean action(int servo_index, float value){
    if (!(rotation_range[0] <= value && value <= rotation_range[1])){
      println("invalid rotation, range is", rotation_range[0], "-", rotation_range[1]);
      return false;
    }
    float median = (this.rotation_range[0] + this.rotation_range[1]) / 2;
    float value_percent = (median - value) / (rotation_range[1] - median);
    float angle = (PI / 2) * value_percent;
    this.pieces[servo_index].move(angle);
    return true;
  }
  
}
