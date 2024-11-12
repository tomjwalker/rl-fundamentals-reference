- [ ] More user-friendly repo setup
    - [x] Add `rl` to PythonPath (+2)
    - [ ] Replace requirements.txt with setup.py
- [ ] Meaningful assignment Q&As (+1)
    - [ ] Add links to the correct solutions in the reference repo
- [ ] Jack's Car Rental background:
    - [x] MVP: attach slides and a bit of blurb to the Assignment
    - [ ] Record video of the slides
- [ ] Mark feedback 12/11/24
    - [ ] Beginner repo, link broken; https://github.com/Curbar-AI/rl-fundamentals-assignments-beginner/blob/main/assignments/mdps.md
    - [ ] 3x3 MDP: frozen hole
    - [ ] https://arc.net/l/quote/lmdpkgqj this bit needs to say that we need to actually implement policy iteration!
    - [ ] and the lecture referred to here: https://arc.net/l/quote/ptaucbeq doesn’t seem to exist
    - [ ] JCP: theta = 1e-8 doesn’t terminate for some reason, 1e-3 is fine and makes the plots like in the readme - does it terminate with 1e-8 in your reference repo?
    - [ ] also also, testing for policy_stable at the end of the outer loop of policy_iteration is kind of redundant because it doesn’t return from policy_improvement until it’s true anyway, maybe it’s not supposed to iterate inside policy_improvement?
    


