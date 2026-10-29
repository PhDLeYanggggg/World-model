# Why A Fixed-Head Veto Cannot Repair This Complete Support Gate

This is an elementary diagnostic for a restricted policy class, not a new
general impossibility theorem or a population-risk guarantee.

The fresh known-only mixed policy has 62/444 required recording views with
undefined selected easy risk. Under the existing finite-completion contract,
that means zero observed selected easy-reference mass, not evidence that
there are no real easy cases. All costs and unknown labels stay unchanged.

For each known query i, the reference R_i and easy-reference ER_i concern the
fixed floor and fixed easy event, not the action k. They are the same across
the seven candidates. If a proposed policy selects only a subset of the
known-only policy's selected rows, then

```text
sum_i selected_new_i * ER_i <= sum_i selected_old_i * ER_i.
```

When the right side is zero, a veto cannot make the left side positive.
Changing which candidate is used on the retained row does not change ER_i.
The same conclusion covers restricting to the old predicted gate-passing
actions: every row with any such action already has an old selected action.
Consequently at least these 62 views remain unsupported, and at most 382/444
can have defined selected easy risk. This ceiling is not an achievable safety
score; known violations or unknown-outcome bounds may further reduce it.

This also rules out one tempting extra training run with otherwise frozen
conditional heads. For any past-only estimated support probability q in [0,1],
the factorized completion proxies would be

```text
B_lower = q*B; H_upper = q*H + (1-q)*E
R_lower = q*R; ER_lower = q*ER; EH_upper = q*EH + (1-q)*E.
```

At q=0 the original gate rejects zero gain/reference. At q>0, passing the
same 2% all/easy upper-harm/lower-reference tests implies H<=0.02R and
EH<=0.02ER; positive conservative gain implies B>H. Thus the action already
passes the frozen conditional head's original gate. Even a perfectly learned
q cannot create the missing support within this policy class.

Do not spend another training grid on a fixed-head availability veto as a
claimed complete repair. A useful next change must improve conditional risk
estimation enough to admit additional causally supported rows, change the
forecast inputs/outputs with evidence, or obtain additional lawful labels.
It must not select easy anchors using observed future labels at inference,
lower the risk threshold or remove the unsupported recordings.

This statement does not rule out jointly retrained conditional heads,
different inputs, different candidate forecasters or M3W in general. It does
not authorize a new evaluation protocol. Independent roles, deployment,
Stage5C and SMC remain unchanged.
