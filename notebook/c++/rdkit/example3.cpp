
#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>

int main(int argc, char **argv) {
 std::string mol_file = "../input_GMX.mol";  //ファイルパス

 // sanitize: true, removeHs: false, strictParsing: true 
 std::shared_ptr<RDKit::ROMol> mol2(RDKit::MolFileToMol(mol_file, true,false,true));  //分子の構築
 // RDKit::MolOps::Kekulize( *mol1, false );

 
 //分子情報の出力std::cout << "Number of atoms(atomic number > 1) " << mol2->getNumAtoms(true) << std::endl;  //水素以外の原子数
 std::cout << "Number of atoms " << mol2->getNumAtoms(false) << std::endl;  //水素を含めた原子数
 std::cout << "Number of heavy atoms " << mol2->getNumHeavyAtoms() << std::endl; //水素以外の原子数
 std::cout << "Number of bonds(atomic number > 1) " << mol2->getNumBonds(true) << std::endl;  //水素が関与しない結合数
 std::cout << "Number of bonds " << mol2->getNumBonds(false) << std::endl;  //水素の関与も含めた総結合数

 //分子情報の出力
 std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
 std::cout << "Number of atoms " << mol2->getNumAtoms(false) << std::endl;  //水素を含めた原子数
 for(auto bond: mol2->bonds()) {
   std::cout << bond->getBondType() << " ";
 }
 std::cout << std::endl;
 // ボンドindexのリストを取得
 for( unsigned int i = 0 , is = mol2->getNumBonds(false) ; i < is ; ++i ) {
   std::cout << "bond index :: " << i << " ";
   const RDKit::Bond *bond = mol2->getBondWithIdx( i ); 
   std::cout << "bond atoms :: " << bond->getBeginAtomIdx() << " " << bond->getEndAtomIdx() << std::endl;
 }
 std::cout << std::endl;
 // 原子番号のリストを取得(これが使える!!)
 for(auto atom: mol2->atoms()) {
   std::cout << atom->getAtomicNum() << " ";
 }
 std::cout << std::endl;

  // 原子番号のリストを取得(これが使える!!)
  std::cout << "原子座標" << std::endl;
  RDKit::Conformer &conf = mol2->getConformer();
  for(int indx=0; indx<mol2->getNumAtoms(false);indx++){
    std::cout << conf.getAtomPos(indx) << std::endl; 
  }
//  for(auto atom: mol2->atoms()) {
//    std::cout << atom->GetAtomPosition() << " ";
//  }
 std::cout << std::endl;

        // # atom list（原子番号）
        // self.atom_list=[]
        // for atom in mol_rdkit.GetAtoms():
        //     self.atom_list.append(atom.GetSymbol())
        
        // #bonds_listの作成
        // self.bonds_list=[]
        // self.double_bonds=[]
        
        // for i,b in enumerate(mol_rdkit.GetBonds()):
        //     indx0 = b.GetBeginAtomIdx()
        //     indx1 = b.GetEndAtomIdx()
        //     bond_type = b.GetBondType()
        
        //     self.bonds_list.append([indx0,indx1])
        //     if str(bond_type) == "DOUBLE" :
        //         self.double_bonds.append(i)
 
 return 0;
}
