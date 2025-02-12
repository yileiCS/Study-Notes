# GitHub Foundations
[GitHub Foundations](https://learn.microsoft.com/en-us/collections/o1njfe825p602p)

## What is Version Control?
**Version control system (VCS)**: 
- a program or set of programs that tracks changes to a collection of files.
- recall earlier versions of individual files or of the entire project
- allow team members to work on a project, even on the same files, without affecting each other's work
- **software configuration management (SCM)** system

### Distributed Version Control
- distirbuted: means that a project's complete history is stored both on the *client* and on the *server*

## Git Terminology
| Terminology                        | Meaning                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Working tree**                   | The set of nested directories and files that contain the project that's being worked on                                                                                                                                                                                                                                                                                                   |
| **Repository (repo)**              | Located at the top level of a working tree, where Git keeps all the history and metadata for a project                                                                                                                                                                                                                                                                                    |
| **Hash**                           | A number produced by a hash function that represents the contents of a file or another object as a fixed number of digits                                                                                                                                                                                                                                                                 |
| **Object**                         | A Git repo contains 4 types of objects: </br> A **blob** object contains an ordinary file; </br> A **tree** object represents a directory; it contains names, hashes, and permissions; </br> A **commit** object represents a specific version of the working tree; </br> A **tag** is a name attached to a commit.                                                                       |
| **Commit**                         | Commits to a database ➡️ you are committing the changes you have made so that others can eventually see them                                                                                                                                                                                                                                                                               |
| **Branch**                         | A named series of linked commits. </br> **Head**: The most recent commit on a branch </br> **Main**: The default branch, which is created when initialize a repository, often named **master* in Git </br> **HEAD**: The head of the current branch </br> Branches allow developers to work independently (or together) in branches and later merge their changes into the default branch |
| **Remote**                         | A named reference to another Git repository. When you create a repo, Git creates a remote named `origin` that is the default remote for push and pull operations                                                                                                                                                                                                                          |
| **Commands, subcommands, options** | Git operations are performed by using commands like `git push` and `git pull`                                                                                                                                                                                                                                                                                                             |


## Git and GitHub
| Git/GitHub | Explanation                                           |
| ---------- | ----------------------------------------------------- |
| **Git**    | a distributed version control system (DVCS)           |
| **GitHub** | a cloud platform that uses Git as its core technology |
