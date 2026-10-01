set pagination off
set confirm off
set breakpoint pending on
set $checkpoints = 0
break SchedulerAccountingCheckpoint
commands
  silent
  if scheduler_under_test->queued_batch_size_ != expected_queued_batch_size
    printf "stage %d: expected batch size %lu, got %lu\n", accounting_stage, expected_queued_batch_size, scheduler_under_test->queued_batch_size_
    quit 1
  end
  if scheduler_under_test->queue_.size_ != expected_request_count
    printf "stage %d: unexpected request count\n", accounting_stage
    quit 1
  end
  if accounting_stage == 3
    if scheduler_under_test->queued_batch_size_ >= 8
      printf "Rejected requests incorrectly satisfy preferred batch size 8\n"
      quit 1
    end
  end
  set $checkpoints = $checkpoints + 1
  continue
end
run
if $_exitcode != 0
  quit 1
end
if $checkpoints != 3
  printf "Expected rejection, drain, and later-admission checkpoints\n"
  quit 1
end
printf "Accounting regression passed at all three checkpoints\n"
