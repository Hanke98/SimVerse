#pragma once

#include <chrono>
#include <string>

namespace dyno
{
	class NewTimer
	{
	public:
		NewTimer();

		void start();
		void stop();
		void reset();

		bool isRunning() const;

		double elapsedMilliseconds() const;
		double elapsedMicroseconds() const;

		void printElapsed(const std::string& label = "") const;
		void printLap(const std::string& label = "");

	private:
		using Clock = std::chrono::high_resolution_clock;
		using TimePoint = Clock::time_point;

		TimePoint mStart;
		TimePoint mLastLap;
		TimePoint mEnd;
		bool mRunning;
	};
}
