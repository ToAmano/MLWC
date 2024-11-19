#include <iostream>
#include <boost/config.hpp>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>// #include <GraphMol/FileParsers/MolSupplier.h>
// #include <GraphMol/FileParsers/MolWriters.h>
// #include <GraphMol/SmilesParse/SmilesParse.h>

int main(int argc, char **argv) {
  std::cout << "hello world" << std::endl;

  using namespace RDKit;
  std::string file_root = getenv("RDBASE");
  file_root += "/Docs/Book";

  std::string mol_file = file_root + "/data/input.mol";
  std::shared_ptr<ROMol> mol2(RDKit::MolFileToMol(mol_file));
}

