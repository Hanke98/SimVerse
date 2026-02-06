#include "NewTimer.h"

#include <iostream>

namespace dyno
{
	NewTimer::NewTimer()
		: mRunning(false)
	{
		reset();
	}

	void NewTimer::start()
	{
		mStart = Clock::now();
		mLastLap = mStart;
		mRunning = true;
	}

	void NewTimer::stop()
	{
		mEnd = Clock::now();
		mRunning = false;
	}

	void NewTimer::reset()
	{
		mStart = Clock::now();
		mLastLap = mStart;
		mEnd = mStart;
		mRunning = false;
	}

	bool NewTimer::isRunning() const
	{
		return mRunning;
	}

	double NewTimer::elapsedMilliseconds() const
	{
		TimePoint endTime = mRunning ? Clock::now() : mEnd;
		return std::chrono::duration<double, std::milli>(endTime - mStart).count();
	}

	double NewTimer::elapsedMicroseconds() const
	{
		TimePoint endTime = mRunning ? Clock::now() : mEnd;
		return std::chrono::duration<double, std::micro>(endTime - mStart).count();
	}

	void NewTimer::printElapsed(const std::string& label) const
	{
		if (!label.empty())
			std::cout << label << ": ";
		std::cout << elapsedMilliseconds() << " ms" << std::endl;
	}

	void NewTimer::printLap(const std::string& label)
	{
		TimePoint now = Clock::now();
		double lapMs = std::chrono::duration<double, std::milli>(now - mLastLap).count();
		mLastLap = now;

		if (!label.empty())
			std::cout << label << ": ";
		std::cout << lapMs << " ms" << std::endl;
	}
}
