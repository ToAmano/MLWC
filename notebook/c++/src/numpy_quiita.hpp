#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>

using namespace std;

namespace {
	template <typename ... Args>
	std::string FormatString(const std::string& fmt, Args ... args)
	{
		size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args ...);
		std::vector<char> buf(len + 1);
		std::snprintf(&buf[0], len + 1, fmt.c_str(), args ...);
		return std::string(&buf[0], &buf[0] + len);
	}
}

bool LoadNpy(const std::string& filePath, std::vector<float>& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f4'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d,)", &axis1);
				
				if (axis1 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		
		fs.read((char*)&data[0], sizeof(float) * axis1);
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<float> >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f4'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d)", &axis1, &axis2);
				
				if (axis1 > 0 && axis2 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
		}
		
		for (int i = 0; i < axis1; i++) {
			fs.read((char*)&data[i][0], sizeof(float) * axis2);
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<std::vector<float> > >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f4'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1, axis3 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d, %d)", &axis1, &axis2, &axis3);
				
				if (axis1 > 0 && axis2 > 0 && axis3 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
			for (int j = 0; j < axis2; j++) {
				data[i][j].resize(axis3);
			}
		}
		
		for (int i = 0; i < axis1; i++) {
			for (int j = 0; j < axis2; j++) {
				fs.read((char*)&data[i][j][0], sizeof(float) * axis3);
			}
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<std::vector<std::vector<float> > > >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f4'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1, axis3 = -1, axis4 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d, %d, %d)", &axis1, &axis2, &axis3, &axis4);
				
				if (axis1 > 0 && axis2 > 0 && axis3 > 0 && axis4 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
			for (int j = 0; j < axis2; j++) {
				data[i][j].resize(axis3);
				for (int k = 0; k < axis3; k++) {
					data[i][j][k].resize(axis4);
				}
			}
		}
		
		for (int i = 0; i < axis1; i++) {
			for (int j = 0; j < axis2; j++) {
				for (int k = 0; k < axis3; k++) {
					fs.read((char*)&data[i][j][k][0], sizeof(float) * axis4);
				}
			}
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<double>& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f8'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d,)", &axis1);
				
				if (axis1 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		
		fs.read((char*)&data[0], sizeof(double) * axis1);
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<double> >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f8'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d)", &axis1, &axis2);
				
				if (axis1 > 0 && axis2 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
		}
		
		for (int i = 0; i < axis1; i++) {
			fs.read((char*)&data[i][0], sizeof(double) * axis2);
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<std::vector<double> > >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f8'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1, axis3 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d, %d)", &axis1, &axis2, &axis3);
				
				if (axis1 > 0 && axis2 > 0 && axis3 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
			for (int j = 0; j < axis2; j++) {
				data[i][j].resize(axis3);
			}
		}
		
		for (int i = 0; i < axis1; i++) {
			for (int j = 0; j < axis2; j++) {
				fs.read((char*)&data[i][j][0], sizeof(double) * axis3);
			}
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool LoadNpy(const std::string& filePath, std::vector<std::vector<std::vector<std::vector<double> > > >& data)
{
	std::ifstream fs(filePath.c_str(), std::ios::in | std::ios::binary);
	if (fs.fail()) { return false; }
	
	// header 6 byte = 0x93NUMPY
	unsigned char magicString[6];
	fs.read((char*)&magicString, sizeof(unsigned char) * 6);
	if ((unsigned char)(magicString[0]) != (unsigned char)0x93 ||
		magicString[1] != 'N' || magicString[2] != 'U' ||
		magicString[3] != 'M' || magicString[4] != 'P' ||
		magicString[5] != 'Y') {
		std::cout << "[ERROR] Not NPY file" << std::endl;
		return false;
	}
	
	unsigned char major, minor;
	fs.read((char*)&major, sizeof(char) * 1);
	fs.read((char*)&minor, sizeof(char) * 1);
	
	unsigned short headerLength;
	fs.read((char*)&headerLength, sizeof(unsigned short));
	
	char* header = new char[headerLength];
	fs.read((char*)header, sizeof(char) * headerLength);
	
	const std::string headerString(header);
	
	bool checkType = false;
	{
		const size_t pos = headerString.find("'descr': ");
		if (pos != std::string::npos) {
			const size_t typePos = headerString.find("'<f8'", pos);
			if (typePos != std::string::npos) {
				checkType = true;
			} else {
				std::cout << "[ERROR] Type Error." << std::endl;
			}
		}
	}
	
	bool checkAxis = false;
	int axis1 = -1, axis2 = -1, axis3 = -1, axis4 = -1;
	{
		const size_t pos = headerString.find("'shape': ");
		if (pos != std::string::npos) {
			const size_t shapeStartPos = headerString.find("(", pos);
			const size_t shapeEndPos   = headerString.find(")", pos);
			
			if (shapeStartPos != std::string::npos && shapeEndPos != std::string::npos) {
				const std::string shapeString = headerString.substr(shapeStartPos, (shapeEndPos - shapeStartPos) + 1);
				::sscanf_s(shapeString.c_str(), "(%d, %d, %d, %d)", &axis1, &axis2, &axis3, &axis4);
				
				if (axis1 > 0 && axis2 > 0 && axis3 > 0 && axis4 > 0) {
					checkAxis = true;
				} else {
					std::cout << "[ERROR] Axis Error." << std::endl;
				}
			}
		}
	}
	
	if (checkType && checkAxis) {
		data.resize(axis1);
		for (int i = 0; i < axis1; i++) {
			data[i].resize(axis2);
			for (int j = 0; j < axis2; j++) {
				data[i][j].resize(axis3);
				for (int k = 0; k < axis3; k++) {
					data[i][j][k].resize(axis4);
				}
			}
		}
		
		for (int i = 0; i < axis1; i++) {
			for (int j = 0; j < axis2; j++) {
				for (int k = 0; k < axis3; k++) {
					fs.read((char*)&data[i][j][k][0], sizeof(double) * axis4);
				}
			}
		}
	}
	
	delete header;
	fs.close();
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<float>& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	
	const std::string headerString = FormatString("{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", axis1);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	fs.write((char*)&data[0], sizeof(float) * axis1);
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<float> >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	
	const std::string headerString = FormatString("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", axis1, axis2);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		fs.write((char*)&data[i][0], sizeof(float) * axis2);
	}
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<std::vector<float> > >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	const int axis3 = (int)data[0][0].size();
	
	const std::string headerString = FormatString("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }", axis1, axis2, axis3);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		for (int j = 0; j < axis2; j++) {
			fs.write((char*)&data[i][j][0], sizeof(float) * axis3);
		}
	}
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<std::vector<std::vector<float> > > >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	const int axis3 = (int)data[0][0].size();
	const int axis4 = (int)data[0][0][0].size();
	
	const std::string headerString = FormatString("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d, %d), }", axis1, axis2, axis3, axis4);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		for (int j = 0; j < axis2; j++) {
			for (int k = 0; k < axis3; k++) {
				fs.write((char*)&data[i][j][k][0], sizeof(float) * axis4);
			}
		}
	}
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<double>& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	
	const std::string headerString = FormatString("{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }", axis1);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	fs.write((char*)&data[0], sizeof(double) * axis1);
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<double> >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	
	const std::string headerString = FormatString("{'descr': '<f8', 'fortran_order': False, 'shape': (%d, %d), }", axis1, axis2);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		fs.write((char*)&data[i][0], sizeof(double) * axis2);
	}
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<std::vector<double> > >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	const int axis3 = (int)data[0][0].size();
	
	const std::string headerString = FormatString("{'descr': '<f8', 'fortran_order': False, 'shape': (%d, %d, %d), }", axis1, axis2, axis3);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		for (int j = 0; j < axis2; j++) {
			fs.write((char*)&data[i][j][0], sizeof(double) * axis3);
		}
	}
	
	return true;
}

bool SaveNpy(const std::string& filePath, const std::vector<std::vector<std::vector<std::vector<double> > > >& data)
{
	std::ofstream fs(filePath.c_str(), std::ios::binary);
	
	const unsigned char magicString[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
	fs.write((char*)magicString, sizeof(magicString));
	
	const unsigned char major = 1, minor = 0;
	fs.write((char*)&major, sizeof(char));
	fs.write((char*)&minor, sizeof(char));
	
	unsigned short headerLength = 118;
	fs.write((char*)&headerLength, sizeof(unsigned short));
	
	const int axis1 = (int)data.size();
	const int axis2 = (int)data[0].size();
	const int axis3 = (int)data[0][0].size();
	const int axis4 = (int)data[0][0][0].size();
	
	const std::string headerString = FormatString("{'descr': '<f8', 'fortran_order': False, 'shape': (%d, %d, %d, %d), }", axis1, axis2, axis3, axis4);
	fs.write((const char*)headerString.c_str(), sizeof(char) * headerString.size());
	
	const int headerSpaceLength = headerLength - (int)headerString.size() - 1;
	if (headerSpaceLength > 0) {
		const std::string headerTailString = std::string(headerSpaceLength, ' ');
		fs.write((const char*)headerTailString.c_str(), sizeof(char) * headerTailString.size());
		const char lineFeed = 0x0A;
		fs.write((char*)&lineFeed, sizeof(char));
	}
	
	for (int i = 0; i < axis1; i++) {
		for (int j = 0; j < axis2; j++) {
			for (int k = 0; k < axis3; k++) {
				fs.write((char*)&data[i][j][k][0], sizeof(double) * axis4);
			}
		}
	}
	
	return true;
}