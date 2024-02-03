// Stopwatch.hpp
// https://takap-tech.com/entry/2019/05/13/235416
#pragma once
#pragma execution_character_set("utf-8")

#include <chrono>
#include <memory>

namespace diagnostics
{
/**
 * C#の System.Stopwatch クラスっぽい動きをする時間測定クラス
 */
class Stopwatch
{
    // 動作しているかのフラグ
    // true : 動作中 / false : 停止中
    bool isRunning = false;
    // 測定開始時刻
    std::chrono::system_clock::time_point begin;
    // 測定開始時刻
    std::chrono::system_clock::time_point end;
    // 合計時間
    std::chrono::nanoseconds elapsed;

    Stopwatch() = default; // CreateNew or StartNewで作って！

public:

    // 意味ないと思うのでmove以外禁止
    ~Stopwatch() = default;
    Stopwatch(const Stopwatch&) = delete;
    Stopwatch& operator=(const Stopwatch) = delete;
    Stopwatch(Stopwatch&&) = default;

    Stopwatch& operator=(Stopwatch&&) = default;

    bool getIsRunning() { return this->isRunning; }

    // 任意の型で計測時間を取り出す
    // Tempalte
    //   Unit  : 取得したい時間の型
    //   Ratio : 時間の単位
    template <typename Unit, class Ratio>
    Unit getElapsed()
    {
        if (this->isRunning)
        {
            return std::chrono::duration_cast<std::chrono::duration<Unit, Ratio>>(
                std::chrono::system_clock::now() - this->begin).count(); // TooLong!!
        }
        else
        {
            return std::chrono::duration_cast<std::chrono::duration<Unit, Ratio>>(this->elapsed).count();
        }
    }

    // 経過時間を秒の倍精度浮動小数点として取り出す
    double getElapsedSeconds()
    {
        return this->getElapsed<double, std::ratio<1>>();
    }

    // 経過時間をミリ秒の倍精度浮動小数点として取り出す
    double getElapsedMilliseconds()
    {
        return this->getElapsed<double, std::milli>();
    }

    // 経過時間をナノ秒として取り出す
    long long getElapsedNanoseconds()
    {
        return this->getElapsed<long long, std::nano>();
    }

    void reset()
    {
        elapsed = std::chrono::nanoseconds(0);
        begin = end = std::chrono::system_clock::now();
    }

    void restart()
    {
        this->reset();
        this->start();
    }

    void start()
    {
        if (this->isRunning)
        {
            return;
        }

        this->isRunning = true;
        begin = end = std::chrono::system_clock::now();
    }

    void stop()
    {
        if (!this->isRunning)
        {
            return;
        }

        this->isRunning = false;
        end = std::chrono::system_clock::now();
        this->elapsed = this->elapsed + std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    }

    static std::unique_ptr<Stopwatch> createNew()
    {
        std::unique_ptr<Stopwatch> sw(new Stopwatch());
        return sw;
    }

    static std::unique_ptr<Stopwatch> startNew()
    {
        auto sw = createNew();
        sw->start();
        return sw;
    }
};
}