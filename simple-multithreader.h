#ifndef SIMPLE_MULTITHREADER_H
#define SIMPLE_MULTITHREADER_H

#include <iostream>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <pthread.h>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <climits>

/* Each worker receives a disjoint half-open interval [begin, end). */
struct SMTWorker {
    long long begin;
    long long end;
    const std::function<void(long long)> *body;
};

/* POSIX thread entry point. The function object remains alive until all
 * workers have been joined by run_parallel(). */
inline void *smt_worker_entry(void *argument) {
    SMTWorker *worker = static_cast<SMTWorker *>(argument);
    for (long long i = worker->begin; i < worker->end; ++i) {
        (*(worker->body))(i);
    }
    return NULL;
}

/* Shared implementation used by both public overloads. Exactly
 * numThreads workers participate: numThreads-1 pthreads plus the caller. */
inline void smt_run_parallel(long long total,
                             const std::function<void(long long)> &body,
                             int numThreads) {
    if (numThreads <= 0) {
        throw std::invalid_argument("parallel_for: numThreads must be positive");
    }
    if (total < 0) {
        throw std::invalid_argument("parallel_for: invalid iteration count");
    }
    if (total == 0) {
        std::cout << "Total execution time: 0 seconds" << std::endl;
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<SMTWorker> workers(static_cast<size_t>(numThreads));
    std::vector<pthread_t> threads(static_cast<size_t>(numThreads > 1 ? numThreads - 1 : 0));
    std::vector<bool> created(static_cast<size_t>(numThreads), false);

    const long long base = total / numThreads;
    const long long remainder = total % numThreads;
    long long next = 0;
    for (int t = 0; t < numThreads; ++t) {
        const long long count = base + (t < remainder ? 1 : 0);
        workers[static_cast<size_t>(t)].begin = next;
        workers[static_cast<size_t>(t)].end = next + count;
        workers[static_cast<size_t>(t)].body = &body;
        next += count;
    }

    /* Start all workers except worker zero, which is executed by the caller. */
    for (int t = 1; t < numThreads; ++t) {
        const int rc = pthread_create(&threads[static_cast<size_t>(t - 1)],
                                      NULL, smt_worker_entry,
                                      &workers[static_cast<size_t>(t)]);
        if (rc == 0) {
            created[static_cast<size_t>(t)] = true;
        } else {
            std::fprintf(stderr, "parallel_for: pthread_create failed (error %d); "
                                 "executing that partition on the calling thread.\n", rc);
        }
    }

    /* The main/calling thread is part of the requested worker count. */
    smt_worker_entry(&workers[0]);

    /* A partition whose pthread could not be created is safely run here. */
    for (int t = 1; t < numThreads; ++t) {
        if (!created[static_cast<size_t>(t)]) {
            smt_worker_entry(&workers[static_cast<size_t>(t)]);
        }
    }

    /* Join every successfully created pthread before returning. */
    for (int t = 1; t < numThreads; ++t) {
        if (created[static_cast<size_t>(t)]) {
            const int rc = pthread_join(threads[static_cast<size_t>(t - 1)], NULL);
            if (rc != 0) {
                std::fprintf(stderr, "parallel_for: pthread_join failed (error %d).\n", rc);
                throw std::runtime_error("parallel_for: unable to join worker thread");
            }
        }
    }

    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
}

/* One-dimensional parallel loop: high is exclusive. */
inline void parallel_for(int low, int high,
                         std::function<void(int)> &&lambda,
                         int numThreads) {
    if (high < low) {
        throw std::invalid_argument("parallel_for: high must be >= low");
    }
    const long long total = static_cast<long long>(high) - low;
    std::function<void(long long)> body =
        [low, &lambda](long long offset) { lambda(static_cast<int>(low + offset)); };
    smt_run_parallel(total, body, numThreads);
}

/* Two-dimensional loop. The iteration space is flattened into a single
 * balanced range and mapped back to (i,j), preserving row-major loop order. */
inline void parallel_for(int low1, int high1, int low2, int high2,
                         std::function<void(int, int)> &&lambda,
                         int numThreads) {
    if (high1 < low1 || high2 < low2) {
        throw std::invalid_argument("parallel_for: upper bounds must be >= lower bounds");
    }
    const long long rows = static_cast<long long>(high1) - low1;
    const long long cols = static_cast<long long>(high2) - low2;
    if (rows != 0 && cols > LLONG_MAX / rows) {
        throw std::overflow_error("parallel_for: iteration space is too large");
    }
    const long long total = rows * cols;
    std::function<void(long long)> body = [=, &lambda](long long index) {
        const long long rowOffset = index / cols;
        const long long colOffset = index % cols;
        lambda(static_cast<int>(low1 + rowOffset),
               static_cast<int>(low2 + colOffset));
    };
    smt_run_parallel(total, body, numThreads);
}

/* The wrapper below lets the supplied sample programs define main() as
 * user_main(), while this header retains control for its demonstration. */
int user_main(int argc, char **argv);

inline void demonstration(std::function<void()> &&lambda) { lambda(); }

int main(int argc, char **argv) {
    int x = 5, y = 1;
    auto lambda1 = [x, &y](void) {
        y = 5;
        std::cout << "====== Welcome to Assignment-" << y
                  << " of the CSE231(A) ======" << std::endl;
    };
    demonstration(lambda1);
    int rc = user_main(argc, argv);
    auto lambda2 = []() {
        std::cout << "====== Hope you enjoyed CSE231(A) ======" << std::endl;
    };
    demonstration(lambda2);
    return rc;
}

#define main user_main
#endif
