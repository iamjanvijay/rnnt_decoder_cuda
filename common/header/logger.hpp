#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#pragma once

/*
 This providea an inteface for logging primitive variables, objects etc into json format.
 TBD:: make it asynchronous and flush all logs into file on kill signal
**/

#include<vector>
#include<string>
#include<iostream>

#ifdef DEBUG
    #define log_d(name, val) s2t::common::log(name, val)
    #define log_d_i(msg) s2t::common::log("info", msg)
#else
    #define log_d(name, val) void()
    #define log_d_i(msg) void()
#endif

#define log_e(name, val) s2t::common::log(name, val)
#define log_i(msg) s2t::common::log("info", msg)

namespace s2t
{
    namespace common
    {
        template<typename T>
        void log(const std::string& name, const T& value)
        {
            std::ostream out(std::cout.rdbuf());
            out<<name<<" : "<<value<<std::endl;
        }

        template<>
        void log<std::vector<size_t>>(const std::string& name, const std::vector<size_t>& obj);
    }
}

#endif