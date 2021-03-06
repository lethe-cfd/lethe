#!/usr/bin/env txr
;;; Copyright (C) 2021 The Lethe Authors
;;;
;;; SPDX-License-Identifier: LGPL-3.0-only

(define-option-struct opts nil
  (n dry-run :bool "Don't copy any files, just show what would be done.")
  (h help :bool "Print this help message and exit.")
  (v verbose :bool "Print the copy commands as they are done."))

(defvarl prog-name (base-name self-path))

(defun usage (: (stream *stdout*))
  (put-line `Usage: @{prog-name} [OPTION...] BUILD-DIR [LETHE-DIR]` stream))

(defun get-dest (path build-dir lethe-dir)
  (match-case path
    (`@{build-dir}/applications_tests/@app/@test.release/@mpirun/output`
     `@{lethe-dir}/applications_tests/@app/@test.@mpirun.output`)
    (`@{build-dir}/applications_tests/@app/@test.release/output`
     `@{lethe-dir}/applications_tests/@app/@test.output`)
    (@otherwise nil)))

(let ((o (new opts)))
  o.(getopts *args*)
  (when o.help
    (put-line "Update Lethe's golden files.\n")
    (usage)
    (put-line "\nIf LETHE-DIR is not provided, it defaults to \"BUILD-DIR/..\".")
    o.(opthelp)
    (put-line `Example: @{prog-name} build`)
    (exit t))

  (let* ((normalize-dir (opip [andf [chain len plusp] identity]
                              (trim-right #/\/+/)))
         (build-dir [normalize-dir (first o.out-args)])
         (lethe-dir (or [normalize-dir (second o.out-args)]
                        (and build-dir `@{build-dir}/..`)))
         (max-non-opts 2))

    (catch (cond ((not build-dir)
                  (throwf 'opt-error "missing build directory"))
                 ((> (len o.out-args) max-non-opts)
                  (throwf 'opt-error "excess non-option arguments: ~s"
                          (nthcdr max-non-opts o.out-args))))
      (opt-error (msg)
        (put-line `@{prog-name}: error: @msg` *stderr*)
        (usage *stderr*)
        (exit 2)))

    (catch (ftw build-dir
                (do whenlet ((dest (get-dest @1 build-dir lethe-dir)))
                    (when (or o.dry-run o.verbose)
                      (put-line `cp @@1 @dest`))
                    (unless o.dry-run
                      (copy-file @1 dest))))
      (file-error (msg)
        (put-line `@{prog-name}: error: @msg` *stderr*)
        (exit nil)))))
