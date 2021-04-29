# Self notes (or development updates and reminders)

So, the environments that are rendered do not match the pddl description of our dataset. Since the dataset would be really used for experimentation, I think the best solution is, when we start heavy experimentation, we adapt the PDDLGym domains to correspond as much as we can to the dataset ones.

Possible issues:

- Renderer is already developed to work with that description, so we need to change them as well
- PDDLGym requires that the pddl files are in a specific format, and we may end up facing some problems when trying to run the standard ones