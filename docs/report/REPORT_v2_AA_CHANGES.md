# Report v2 (AA): summary of changes

**TL;DR:** `REPORT_v2_AA.md` / `.pdf` apply Albert's markup of the v1 PDF. Process notes and side remarks are cut, eight tables are reformatted, two citations are corrected, and the transaction-cost results are reframed around what the agent learns. No result changed; v1 is untouched.

**Cuts (yellow).** We removed process detail (commit hashes, dates, test count, download rationale, risk-free check, reward-noise footnote). We also cut untested explanations of the spline and $10B results and the "Known gaps" and "Sources" paragraphs. The Section 4 disclosure is now one sentence, "Use of the benchmark"; it is the only disclosure of the benchmark-sized budget.

**Verified (green).** Elminejad et al. now cites the 2025 *Journal of Economic Surveys* version. "Median near 7" for Barsky et al. (1997) was unsupported and is replaced by "most respondents fall in the least risk-tolerant category". Mehra and Prescott's bound of 10 is correct.

**Reformatted (blue).**
- **Tables:** Table 1 has a wider Exposure column. Table 3 has one column per asset in percent. Tables 4 and A3 have rebalanced widths, and Table 4's reasons are trimmed. Table 5 combines seeds and days in one cell. Table 6 adds a median-seed row. Table 9 gives each γ a gap column and a change column. Table 12 is split into a value-gap panel and a trading panel.
- **Text:** The Section 5 opening, the rank-correlation sentence, the conditioning paragraph and the "did not rescue" sentence are rewritten in plainer language.
- **Transaction costs:** the agent is now framed as closing most of the gap to a full-information oracle as costs rise. The lead-in, abstract and conclusion are updated.

**Code and data added.**
- `src/benchmarks/closed_form_table.py` writes the budget-only closed-form weights to `results/benchmarks/`. It checks them against a numerical solve.
- The six transaction-cost oracle tables in `results/tc-drift/oracle/*.npz` (about 1 MB) are committed.
