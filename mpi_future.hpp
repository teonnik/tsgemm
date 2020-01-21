//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LIBS_MPI_FUTURE_HPP
#define HPX_LIBS_MPI_FUTURE_HPP

#include <mpi.h>
//
#include <cstdio>
#include <utility>
#include <tuple>
//
#include <hpx/local_lcos/promise.hpp> 
#include <hpx/memory/intrusive_ptr.hpp>
//

namespace hpx { namespace mpi {

    // -----------------------------------------------------------------
    // An implementation of future_data for MPI
    // -----------------------------------------------------------------
    struct future_data : hpx::lcos::detail::future_data<void>
    {
        HPX_NON_COPYABLE(future_data);

        using init_no_addref = hpx::lcos::detail::future_data<void>::init_no_addref;

        // default empty constructor
        future_data() = default;

        // constructor that takes a request
        future_data(init_no_addref no_addref, MPI_Request request);

        // constructor used for creation directly by invoke
        future_data(init_no_addref no_addref);

        // The native MPI request handle owned by this future data
        MPI_Request request_;
    };

    using future_data_ptr = memory::intrusive_ptr<future_data>;

    // -----------------------------------------------------------------
    // utility function to add a new request to the list to be tracked
    void add_to_request_list(future_data_ptr data);

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    hpx::future<void> get_future(MPI_Request request);

    // -----------------------------------------------------------------
    // return a future from an async call to MPI_Ixxx function
    template <typename F, typename ...Ts>
    hpx::future<void> async(F f, Ts &&...ts)
    {
        future_data_ptr data =
            new future_data(future_data::init_no_addref{});

        // invoke the call to MPI_Ixxx
        f(std::forward<Ts>(ts)..., &data->request_);

        // add the new shared state to the list for tracking
        add_to_request_list(data);

        // return a new future with the mpi::future_data shared state
        using traits::future_access;
        return future_access<hpx::future<void>>::create(std::move(data));
    }

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::future ready
    // when found
    void poll();

    // This is not completely safe as it will return when the request list is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    void wait();
}}

#endif
