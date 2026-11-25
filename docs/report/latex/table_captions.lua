-- Move a "**Table N.** ..." paragraph that sits just before a table into that
-- table's caption, so LaTeX keeps the caption with the table.
function Blocks(blocks)
  local out = pandoc.List()
  local i = 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    if b.t == "Para" and nxt and nxt.t == "Table" and #b.content > 0
       and b.content[1].t == "Strong"
       and pandoc.utils.stringify(b.content[1]):match("^Table %w+%.") then
      nxt.caption.long = { pandoc.Plain(b.content) }
      out:insert(nxt)
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  return out
end
