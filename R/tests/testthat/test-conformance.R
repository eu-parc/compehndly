test_that("shared conformance vectors are available and runnable", {
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("polars")

  if (!exists("compehndly_apply", mode = "function")) {
    skip("R implementation should expose compehndly_apply(function_name, ..., .params = list())")
  }

  cases_path <- conformance_file_path()
  expect_true(file.exists(cases_path))

  payload <- jsonlite::fromJSON(cases_path, simplifyVector = FALSE)
  cases <- payload$cases

  for (case in cases) {
    function_name <- case$function
    params <- case$params %||% list()
    input <- case$input %||% list()

    invoke <- function() {
      if (!is.null(input$positional)) {
        series <- lapply(input$positional, polars::pl$Series)
        return(do.call(compehndly_apply, c(list(function_name), series, .params = list(params))))
      }

      if (!is.null(input$named)) {
        named_series <- lapply(input$named, polars::pl$Series)
        return(do.call(compehndly_apply, c(list(function_name), named_series, .params = list(params))))
      }

      do.call(compehndly_apply, list(function_name, .params = params))
    }

    if (!is.null(case$expect_error)) {
      expect_error(invoke())
      next
    }

    out <- invoke()
    assertions <- case$assertions %||% list()

    if (!is.null(assertions$expected)) {
      got <- to_numeric_with_na(out$to_list())
      expected <- to_numeric_with_na(assertions$expected)
      expect_equal(got, expected, tolerance = 1e-8)
    }

    if (isTRUE(assertions$all_null)) {
      expect_equal(out$null_count(), out$len())
    }

    if (isTRUE(assertions$non_negative)) {
      got <- as.numeric(out$to_numpy())
      expect_true(all(got >= 0))
    }

    if (isTRUE(assertions$no_nan)) {
      got <- as.numeric(out$to_numpy())
      expect_false(any(is.nan(got)))
    }

    if (!is.null(assertions$equals_at_indices)) {
      got <- as.numeric(out$to_numpy())
      for (idx_name in names(assertions$equals_at_indices)) {
        idx <- as.integer(idx_name) + 1
        expect_equal(got[[idx]], as.numeric(assertions$equals_at_indices[[idx_name]]), tolerance = 1e-8)
      }
    }

    if (!is.null(assertions$ranges_at_indices)) {
      got <- as.numeric(out$to_numpy())
      for (idx_name in names(assertions$ranges_at_indices)) {
        idx <- as.integer(idx_name) + 1
        bounds <- assertions$ranges_at_indices[[idx_name]]
        expect_true(got[[idx]] >= as.numeric(bounds[[1]]))
        expect_true(got[[idx]] <= as.numeric(bounds[[2]]))
      }
    }

    if (!is.null(assertions$min_value_from_index)) {
      got <- as.numeric(out$to_numpy())
      start <- as.integer(assertions$min_value_from_index$start) + 1
      min_value <- as.numeric(assertions$min_value_from_index$value)
      expect_true(all(got[start:length(got)] >= min_value))
    }
  }
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
