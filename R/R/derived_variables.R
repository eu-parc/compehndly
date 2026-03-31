`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.stop_if_missing_polars <- function() {
  if (!requireNamespace("polars", quietly = TRUE)) {
    stop("Package 'polars' is required.", call. = FALSE)
  }
}

.series_to_list <- function(x) {
  if (is.null(x)) return(NULL)
  x$to_list()
}

.series_to_numeric <- function(x) {
  vals <- .series_to_list(x)
  vapply(
    vals,
    function(v) {
      if (is.null(v)) return(NA_real_)
      as.numeric(v)
    },
    numeric(1)
  )
}

.series_to_logical <- function(x) {
  vals <- .series_to_list(x)
  vapply(
    vals,
    function(v) {
      if (is.null(v)) return(NA)
      as.logical(v)
    },
    logical(1)
  )
}

.as_scalar_numeric <- function(x, name) {
  if (is.null(x)) {
    stop(sprintf("Parameter '%s' is required", name), call. = FALSE)
  }

  if (is.numeric(x) && length(x) == 1) {
    return(as.numeric(x))
  }

  if (is.list(x) && length(x) == 1) {
    return(as.numeric(x[[1]]))
  }

  if (is.environment(x) && !is.null(x$to_list)) {
    vals <- x$to_list()
    if (length(vals) == 0) {
      stop(sprintf("Parameter '%s' must contain at least one value", name), call. = FALSE)
    }
    first <- vals[[1]]
    if (is.null(first)) {
      stop(sprintf("Parameter '%s' cannot be null", name), call. = FALSE)
    }
    return(as.numeric(first))
  }

  if (length(x) == 1) {
    return(as.numeric(x))
  }

  stop(sprintf("Parameter '%s' must be scalar", name), call. = FALSE)
}

.to_series <- function(values) {
  polars::pl$Series(values)
}

.validate_same_length <- function(...) {
  vectors <- list(...)
  lengths <- vapply(vectors, length, integer(1))
  if (length(unique(lengths)) > 1) {
    stop("All input series must have the same length", call. = FALSE)
  }
}

.fit_censored_lognorm <- function(values_np, censored_np) {
  if (sum(!censored_np) == 0) {
    stop("Cannot fit lognormal: all observations are censored.", call. = FALSE)
  }

  nll <- function(params) {
    sigma <- params[[1]]
    mu <- params[[2]]

    if (sigma <= 0) {
      return(Inf)
    }

    ll_unc <- sum(dlnorm(values_np[!censored_np], meanlog = mu, sdlog = sigma, log = TRUE))
    ll_cens <- sum(log(plnorm(values_np[censored_np], meanlog = mu, sdlog = sigma)))

    penalty <- 0
    if (sigma < 0.05) {
      penalty <- penalty + 1e3 * (0.05 - sigma) ^ 2
    }
    if (sigma > 5.0) {
      penalty <- penalty + 1e3 * (sigma - 5.0) ^ 2
    }

    -(ll_unc + ll_cens - penalty)
  }

  unc <- values_np[!censored_np]
  mu0 <- log(stats::median(unc))
  sigma0 <- stats::sd(log(unc))
  sigma0 <- if (is.na(sigma0) || sigma0 <= 0.1) 0.5 else sigma0

  fit <- stats::optim(
    par = c(sigma0, mu0),
    fn = nll,
    method = "L-BFGS-B",
    lower = c(1e-6, -Inf),
    upper = c(Inf, Inf)
  )

  if (fit$convergence != 0) {
    stop(sprintf("Censored MLE did not converge: %s", fit$message %||% "unknown error"), call. = FALSE)
  }

  list(sigma = fit$par[[1]], mu = fit$par[[2]])
}

.validate_thresholds <- function(loq, lod = NULL) {
  if (loq <= 0) {
    stop("loq must be > 0", call. = FALSE)
  }
  if (!is.null(lod)) {
    if (lod <= 0) {
      stop("lod must be > 0", call. = FALSE)
    }
    if (lod >= loq) {
      stop("lod must be < loq", call. = FALSE)
    }
  }
}

# ---------------- Derived Variable Implementations ----------------

.dv_summation <- function(..., all_required = TRUE) {
  series <- list(...)
  if (length(series) == 0) {
    stop("At least one input is required", call. = FALSE)
  }

  vectors <- lapply(series, .series_to_numeric)
  do.call(.validate_same_length, vectors)

  n <- length(vectors[[1]])

  if (isTRUE(all_required)) {
    any_entirely_null <- any(vapply(vectors, function(v) all(is.na(v)), logical(1)))
    if (any_entirely_null) {
      return(.to_series(rep(NA_real_, n)))
    }
  }

  mat <- do.call(cbind, lapply(vectors, function(v) ifelse(is.na(v), 0, v)))
  out <- rowSums(mat)
  .to_series(out)
}

.dv_standardize <- function(measured, standard) {
  .to_series(.series_to_numeric(measured) * 100 / .series_to_numeric(standard))
}

.dv_standardize_creatinine <- function(measured, crt) {
  .dv_standardize(measured = measured, standard = crt)
}

.dv_normalize_specific_gravity <- function(measured, sg_measured, sg_ref) {
  sg_ref_num <- .as_scalar_numeric(sg_ref, "sg_ref")
  .to_series(.series_to_numeric(measured) * (sg_ref_num - 1) / .series_to_numeric(sg_measured))
}

.dv_total_lipid_concentration <- function(chol, trigl) {
  .to_series((.series_to_numeric(chol) * 2.27) + .series_to_numeric(trigl) + 62.3)
}

.dv_standardize_lipid <- function(measured, lipid_value) {
  .dv_standardize(measured = measured, standard = lipid_value)
}

.dv_medium_bound_imputation_scalar_input <- function(measurement, loq, lod = NULL) {
  loq_num <- .as_scalar_numeric(loq, "loq")
  lod_num <- if (is.null(lod)) NULL else .as_scalar_numeric(lod, "lod")
  .validate_thresholds(loq_num, lod_num)

  m <- .series_to_numeric(measurement)
  out <- m

  if (is.null(lod_num)) {
    mask <- !is.na(m) & (m < loq_num)
    out[mask] <- loq_num / 2
    return(.to_series(out))
  }

  mask_below_lod <- !is.na(m) & (m < lod_num)
  out[mask_below_lod] <- lod_num / 2

  midpoint <- (lod_num + loq_num) / 2
  mask_between <- !is.na(m) & (m >= lod_num) & (m < loq_num)
  out[mask_between] <- midpoint

  .to_series(out)
}

.dv_medium_bound_imputation <- function(measurement, loq, lod = NULL) {
  m <- .series_to_numeric(measurement)
  loq_v <- .series_to_numeric(loq)
  lod_v <- if (is.null(lod)) NULL else .series_to_numeric(lod)

  .validate_same_length(m, loq_v)
  if (!is.null(lod_v)) .validate_same_length(m, lod_v)

  out <- m

  if (is.null(lod_v)) {
    mask <- !is.na(m) & !is.na(loq_v) & (m < loq_v)
    out[mask] <- loq_v[mask] / 2
    return(.to_series(out))
  }

  mask_below_lod <- !is.na(m) & !is.na(lod_v) & (m < lod_v)
  out[mask_below_lod] <- lod_v[mask_below_lod] / 2

  midpoint <- (lod_v + loq_v) / 2
  mask_between <- !is.na(m) & !is.na(lod_v) & !is.na(loq_v) & (m >= lod_v) & (m < loq_v)
  out[mask_between] <- midpoint[mask_between]

  .to_series(out)
}

.dv_random_single_imputation <- function(biomarker, lod, loq, seed = NULL) {
  lod_num <- .as_scalar_numeric(lod, "lod")
  loq_num <- .as_scalar_numeric(loq, "loq")
  .validate_thresholds(loq_num, lod_num)

  biomarker_np <- .series_to_numeric(biomarker)
  biomarker_filled <- ifelse(is.na(biomarker_np), -1, biomarker_np)

  censored <- biomarker_filled < 0
  values_np <- ifelse(censored, lod_num, biomarker_filled)

  fit <- .fit_censored_lognorm(values_np, censored)

  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }

  lower <- rep(0, length(biomarker_filled))
  upper <- rep(0, length(biomarker_filled))

  cat_below_lod <- biomarker_filled == -1
  cat_between <- biomarker_filled == -2
  cat_below_loq <- biomarker_filled == -3

  lower[cat_below_lod] <- 0
  upper[cat_below_lod] <- lod_num

  lower[cat_between] <- lod_num
  upper[cat_between] <- loq_num

  lower[cat_below_loq] <- 0
  upper[cat_below_loq] <- loq_num

  cdf_lo <- plnorm(lower, meanlog = fit$mu, sdlog = fit$sigma)
  cdf_hi <- plnorm(upper, meanlog = fit$mu, sdlog = fit$sigma)

  u <- stats::runif(length(cdf_lo), min = cdf_lo, max = cdf_hi)
  imputed <- qlnorm(u, meanlog = fit$mu, sdlog = fit$sigma)

  out <- biomarker_filled
  out[censored] <- imputed[censored]

  .to_series(out)
}

.DERIVED_FUNCTIONS <- list(
  summation = .dv_summation,
  standardize = .dv_standardize,
  standardize_creatinine = .dv_standardize_creatinine,
  normalize_specific_gravity = .dv_normalize_specific_gravity,
  total_lipid_concentration = .dv_total_lipid_concentration,
  standardize_lipid = .dv_standardize_lipid,
  medium_bound_imputation_scalar_input = .dv_medium_bound_imputation_scalar_input,
  medium_bound_imputation = .dv_medium_bound_imputation,
  random_single_imputation = .dv_random_single_imputation
)

compehndly_apply <- function(function_name, ..., .params = list()) {
  .stop_if_missing_polars()

  fn <- .DERIVED_FUNCTIONS[[function_name]]
  if (is.null(fn)) {
    stop(sprintf("Unknown function '%s'", function_name), call. = FALSE)
  }

  data_args <- list(...)
  call_args <- c(data_args, .params)
  do.call(fn, call_args)
}
