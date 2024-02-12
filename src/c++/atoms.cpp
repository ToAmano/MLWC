/*
 2023/5/30
 ase atomsに対応するAtomsクラスを定義する
*/

#include <vector>

class AtomicCord // (x,y,z)の三次元ベクトル
{
  public:
    double position[3];
    AtomicCord(double x, double y, double z);  // コンストラクタ
};

AtomicCord::AtomicCord(double x, double y, double z)
{
  position[0] = x;
  position[1] = y;
  position[2] = z;  
  // this->position = {x,y,z};
};


class Atoms {
public: // public変数
  const char* owner;
  const char* colour;
  int number;
  Atoms(std::vector<int> atomic_num,
        std::vector<vector<double> > positions,
        std::vector<std::vectorcell= UNITCELL_VECTORS)
  //pbc=[1, 1, 1]))
};
