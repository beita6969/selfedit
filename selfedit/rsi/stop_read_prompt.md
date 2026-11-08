Readings of the edits measured so far, taken on your own reasoning traces from this round without generating anything (mean over {n} traces). Base log-probability of closing your thinking: {cut} at points late in a trace, {end} where you actually stopped.

stop: change in the log-probability of closing your thinking at those late points. Higher means readier to stop. Random directions read below 0.1.
after the answer: the same change right after lines where you state the correct answer.
end: the same change where you actually stopped.
all: change in the mean log-probability of your own reasoning tokens. Below -{cap} the edit disturbs your reasoning and is rejected.
revisit: change at the words that reopen a line of thought (Wait, Hmm, Actually, Alternatively, But).

{table}

A higher stop helps only if the answer is already right when you end. Using what these readings show, propose up to {max_edits} further edits in the same JSON format. Readings can be taken cheaply, so an untried layer or direction is worth proposing.

Stop readings of edits on different layers add up, while each edit's drift grows with the square of its strength. So several weak edits on different layers reach the same stop at less drift than one strong edit; a combo is one candidate.
