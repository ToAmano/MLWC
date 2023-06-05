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
  this->position = {x,y,z};
};


class Atoms {
public: // public変数
  const char* owner;
  const char* colour;
  int number;
  Atoms(std::vector<int> atomic_num,
        std::vector<AtomicCord.position> positions,
        cell= UNITCELL_VECTORS)
  //pbc=[1, 1, 1]))
};
