/**
 * @file timer.hpp
 * @author your name (you@domain.com)
 * @brief other time related functions
 * @version 0.1
 * @date 2024-02-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#pragma once
#pragma execution_character_set("utf-8")

#include <chrono>
#include <iostream> // std::cout
#include <iomanip>  // std::setw


namespace diel_timer
{
void print_current_time(std::string print_str){
    std::chrono::system_clock::time_point  current_time; // 型は auto で可
    std::time_t time = std::chrono::system_clock::to_time_t(current_time);
    std::cout << std::setw(30) << print_str << std::ctime(&time) << std::endl;
};
}